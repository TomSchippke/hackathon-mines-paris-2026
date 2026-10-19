"""Agent de fusion : regroupe les clusters fins en pathways de remédiation (clusters fusionnés).

Un cluster fusionné = une seule action de remédiation concrète qui traite tous ses membres.
Le DBCV sert de garde-fou (veto sauf confirmation d'un critique) ; un cluster signalé
hétérogène est re-clusterisé de façon déterministe (balayage HDBSCAN).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .clustering import dbcv_score, sweep_hdbscan
from .config import (
    MCS_SWEEP,
    MERGE_DBCV_HARD_DROP,
    MERGE_DBCV_MAX_DROP,
    MERGE_MAX_GROUP,
    MERGE_MAX_ROUNDS,
    MERGE_NEIGHBOURS,
    OUTLIER_LABEL,
)
from .llm_client import call_llm_json

TITLE_JACCARD_MIN = 0.6
SPLIT_MAX_PARTS = 4  # au-delà, HDBSCAN découpe un mode homogène en fragments

SYSTEM_MERGE_PROMPT = (
    "Tu es un ingénieur fiabilité senior. On te présente un groupe de clusters d'incidents "
    "industriels voisins (modes de défaillance). Ta tâche : décider lesquels peuvent être "
    "fusionnés en un seul pathway de remédiation.\n"
    "Critère strict : fusionner uniquement si une même action de remédiation concrète traite "
    "tous les membres. Une simple similarité de symptômes, de vocabulaire ou d'équipement ne "
    "suffit pas. Pour chaque membre fusionné, indique le résultat (défaillance) que cette "
    "action évite.\n"
    "Tu peux aussi signaler (flags) un cluster qui te semble hétérogène (plusieurs modes de "
    "défaillance mélangés) afin qu'il soit re-découpé.\n"
    "Propose de petites fusions (typiquement des paires ou des triplets de clusters qui partagent "
    "la même correction) et non la fusion de tout le groupe : la plupart des groupes ne "
    "contiennent que quelques fusions, voire aucune.\n"
    "Utilise uniquement les identifiants fournis. Un cluster ne peut apparaître que dans une "
    "seule fusion. En cas de doute, ne fusionne pas.\n"
    'Réponds en JSON : {"merges": [{"members": ["id", ...], "shared_remediation": "...", '
    '"prevented_outcomes": [{"fine_cluster_id": "id", "outcome": "..."}], '
    '"justification": "..."}], "flags": [{"fine_cluster_id": "id", "reason": "..."}]}'
)

SYSTEM_CRITIC_PROMPT = (
    "Tu es un relecteur critique en fiabilité industrielle. Une fusion de clusters d'incidents "
    "dégrade nettement la cohérence géométrique des données : tu ne la confirmes que si elle est "
    "indiscutable. Par défaut, rejette.\n"
    "Confirme uniquement si UNE action de remédiation unique, portant sur UN composant ou "
    "sous-système précis, empêche la défaillance de CHAQUE membre.\n"
    "Rejette notamment si :\n"
    "- la remédiation vise deux composants ou systèmes différents, ou enchaîne des actions "
    "distinctes (« et », « ou », « / ») ;\n"
    "- elle est générique (programme de maintenance, inspection régulière, surveillance "
    "renforcée, contrôle systématique...) ;\n"
    "- les mécanismes de défaillance des membres diffèrent au point que l'action n'empêcherait "
    "pas l'un d'eux ;\n"
    "- le lien entre l'action et un membre repose sur une hypothèse non visible dans les "
    "incidents présentés.\n"
    "Nomme le composant et l'action ; laisse-les vides si tu ne peux pas en nommer un seul.\n"
    'Réponds en JSON : {"component": "...", "action": "...", "addresses_all_members": true|false, '
    '"single_specific_remediation": true|false, "reason": "..."}'
)


@dataclass
class FineState:
    """État d'une partition en entrée de l'agent de fusion."""

    partition: str
    X_umap: np.ndarray
    embeddings: np.ndarray
    fine_labels: np.ndarray
    cards: dict[str, dict]
    titles: dict[str, dict]


@dataclass
class MergeResult:
    """Sortie de l'agent de fusion."""

    fine_labels: np.ndarray
    fine_to_merged: dict[str, str]
    merged_members: dict[str, list[str]]
    log: list[dict] = field(default_factory=list)
    titles: dict[str, dict] = field(default_factory=dict)


# --------------------------------------------------------------------------- utilitaires


def _tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _title(titles: dict[str, dict], fid: str) -> str:
    t = titles.get(fid) or {}
    return str(t.get("fine_title") or fid)


def _is_noise(lab: Any) -> bool:
    return lab == OUTLIER_LABEL or lab == -1


class _Node:
    """Nœud de fusion : un cluster fin ou un groupe déjà fusionné."""

    def __init__(self, node_id: str, members: list[str]):
        self.id = node_id
        self.members = members  # ids de clusters fins (aplatis)


def _chunk_group(ids: list[str], centroids: dict[str, np.ndarray], max_size: int) -> list[list[str]]:
    """Découpe glouton d'une composante trop grande en paquets de voisins proches."""
    remaining = list(ids)
    chunks: list[list[str]] = []
    while remaining:
        chunk = [remaining.pop(0)]
        while remaining and len(chunk) < max_size:
            c = np.mean([centroids[i] for i in chunk], axis=0)
            sims = [float(centroids[r] @ c) for r in remaining]
            chunk.append(remaining.pop(int(np.argmax(sims))))
        chunks.append(chunk)
    return chunks


def _candidate_groups(
    ids: list[str],
    centroids: dict[str, np.ndarray],
    node_titles: dict[str, str],
) -> list[list[str]]:
    """Groupes candidats : k plus proches centroïdes + titres quasi identiques, composantes connexes."""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(ids) >= 2:
        M = np.stack([centroids[i] for i in ids])
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        k = min(MERGE_NEIGHBOURS, len(ids) - 1)
        for a, i in enumerate(ids):
            for b in np.argsort(-S[a])[:k]:
                union(i, ids[int(b)])
        toks = {i: _tokens(node_titles[i]) for i in ids}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if _jaccard(toks[ids[a]], toks[ids[b]]) >= TITLE_JACCARD_MIN:
                    union(ids[a], ids[b])

    comps: dict[str, list[str]] = {}
    for i in ids:
        comps.setdefault(find(i), []).append(i)
    groups: list[list[str]] = []
    for comp in comps.values():
        if len(comp) < 2:
            continue
        if len(comp) > MERGE_MAX_GROUP:
            groups.extend(c for c in _chunk_group(comp, centroids, MERGE_MAX_GROUP) if len(c) >= 2)
        else:
            groups.append(comp)
    return groups


PROMPT_MAX_CHARS = 12000


def _node_block(node: _Node, size: int, titles: dict[str, dict], cards: dict[str, dict], rep_chars: int = 250) -> str:
    """Bloc texte d'un nœud. Nœud > 2 membres : 1 exemple par membre, 4 membres détaillés au plus."""
    big = len(node.members) > 2
    lines = [f"### Cluster {node.id} (taille {size})"]
    for k, fid in enumerate(node.members):
        t = titles.get(fid) or {}
        card = cards.get(fid) or {}
        fm = str(t.get("failure_mode", ""))
        if rep_chars < 0:  # mode compact : titres et modes tronqués, ni mots-clés ni exemples
            lines.append(f"- Membre {fid} : {_title(titles, fid)[:60]} | {fm[:50]}")
            continue
        lines.append(f"- Membre {fid} : {_title(titles, fid)} | mode de défaillance : {fm}")
        if big and k >= 4:
            continue
        kws = card.get("keywords") or []
        if kws:
            lines.append(f"  mots-clés : {', '.join(map(str, kws[:8]))}")
        if rep_chars > 0:
            for txt in (card.get("representative_texts") or [])[: 1 if big else 2]:
                lines.append(f"  exemple : {str(txt)[:rep_chars]}")
    return "\n".join(lines)


def _group_prompt(partition: str, alive: list[str], nodes: dict, cur: np.ndarray, titles: dict, cards: dict) -> str:
    """Prompt d'un groupe, borné à PROMPT_MAX_CHARS par réduction des exemples."""
    head = f"Partition : {partition}. Voici {len(alive)} clusters voisins.\n\n"
    tail = "\n\nPropose les fusions et les drapeaux d'hétérogénéité."
    out = ""
    for rep_chars in (250, 120, 60, 0, -1):
        body = "\n\n".join(
            _node_block(nodes[g], int((cur == g).sum()), titles, cards, rep_chars) for g in alive
        )
        out = head + body + tail
        if len(out) <= PROMPT_MAX_CHARS:
            return out
    return out[: PROMPT_MAX_CHARS - len(tail)] + tail


def _parse_merges(resp: Any, group_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Validation défensive de la réponse : ids inconnus ignorés, doublons inter-fusions écartés."""
    if not isinstance(resp, dict):
        return [], []
    merges: list[dict] = []
    used: set[str] = set()
    raw_merges = resp.get("merges")
    for m in raw_merges if isinstance(raw_merges, list) else []:
        if not isinstance(m, dict) or not isinstance(m.get("members"), list):
            continue
        members: list[str] = []
        for x in m["members"]:
            if isinstance(x, str) and x in group_ids and x not in members:
                members.append(x)
        members = [x for x in members if x not in used]
        if len(members) < 2:
            continue
        used.update(members)
        outcomes = []
        for o in m.get("prevented_outcomes") if isinstance(m.get("prevented_outcomes"), list) else []:
            if isinstance(o, dict) and o.get("fine_cluster_id") in members:
                outcomes.append({"fine_cluster_id": o["fine_cluster_id"], "outcome": str(o.get("outcome", ""))})
        merges.append(
            {
                "members": members,
                "shared_remediation": str(m.get("shared_remediation") or ""),
                "prevented_outcomes": outcomes,
                "justification": str(m.get("justification") or ""),
            }
        )
    flags: list[dict] = []
    raw_flags = resp.get("flags")
    for f in raw_flags if isinstance(raw_flags, list) else []:
        if isinstance(f, dict) and isinstance(f.get("fine_cluster_id"), str) and f["fine_cluster_id"] in group_ids:
            if all(f["fine_cluster_id"] != g["fine_cluster_id"] for g in flags):
                flags.append({"fine_cluster_id": f["fine_cluster_id"], "reason": str(f.get("reason") or "")})
    return merges, flags


def _minimal_card(parent_card: dict, size: int) -> dict:
    return {"keywords": list(parent_card.get("keywords") or []), "representative_texts": [], "size": int(size)}


# --------------------------------------------------------------------------- agent


def run_merge_agent(
    state: FineState,
    llm: Callable = call_llm_json,
    relabel: Callable[[np.ndarray, list[str]], tuple[dict[str, dict], dict[str, dict]]] | None = None,
    max_rounds: int = MERGE_MAX_ROUNDS,
) -> MergeResult:
    """Fusionne les clusters fins d'une partition en clusters de remédiation (§T4).

    Les drapeaux d'hétérogénéité ne s'appliquent qu'aux clusters fins non fusionnés.
    Les points de bruit d'un re-découpage sont rattachés au sous-cluster le plus proche.
    """
    partition = state.partition
    X = np.asarray(state.X_umap)
    E = np.asarray(state.embeddings, dtype=float)
    fine = np.array(list(state.fine_labels), dtype=object)  # étiquettes fines (avec scissions)
    cur = fine.copy()  # étiquettes de nœuds (fins ou fusionnés)
    cards = {k: dict(v) for k, v in state.cards.items()}
    titles = {k: dict(v) for k, v in state.titles.items()}

    nodes: dict[str, _Node] = {}
    for lab in dict.fromkeys(cur.tolist()):
        if not _is_noise(lab):
            nodes[lab] = _Node(lab, [lab])

    log: list[dict] = []
    parent_of: dict[str, str] = {}  # nœud fusionné intermédiaire -> nœud qui l'absorbe
    counter = 0
    rejected: list[frozenset[str]] = []  # ensembles de clusters fins déjà refusés : jamais reproposés

    for rnd in range(1, max_rounds + 1):
        ids = sorted(nodes)
        if len(ids) < 2:
            break
        centroids: dict[str, np.ndarray] = {}
        for nid in ids:
            c = E[cur == nid].mean(axis=0)
            centroids[nid] = c / (np.linalg.norm(c) or 1.0)
        node_titles = {nid: " ".join(_title(titles, f) for f in nodes[nid].members) for nid in ids}
        groups = _candidate_groups(ids, centroids, node_titles)
        accepted = 0

        # -------- fusions
        pending_flags: list[dict] = []
        for group in groups:
            alive = [g for g in group if g in nodes]  # un nœud absorbé plus tôt dans le tour est ignoré
            if len(alive) < 2:
                continue
            user = _group_prompt(partition, alive, nodes, cur, titles, cards)
            try:
                resp = llm(SYSTEM_MERGE_PROMPT, user, cache_tag="merge")
            except Exception:
                continue
            merges, flags = _parse_merges(resp, set(alive))
            pending_flags.extend(flags)

            for m in merges:
                members = [x for x in m["members"] if x in nodes]
                if len(members) < 2:
                    continue
                proposed = frozenset(f for x in members for f in nodes[x].members)
                if any(r <= proposed for r in rejected):
                    continue  # reformulation d'une fusion déjà refusée
                before = dbcv_score(X, cur)
                trial = cur.copy()
                counter += 1
                new_id = f"{partition}_Mtmp{counter}"
                for x in members:
                    trial[trial == x] = new_id
                after = dbcv_score(X, trial)
                vetoed, verdict = False, None
                drop = (before - after) / max(abs(before), 1e-9)
                if before > 0 and drop > MERGE_DBCV_HARD_DROP:
                    # Veto ferme : le critique LLM, trop conciliant, ne peut pas l'outrepasser
                    vetoed = True
                    verdict = {"single_specific_remediation": False, "hard_veto": True,
                               "reason": f"chute du DBCV {drop:.0%} > {MERGE_DBCV_HARD_DROP:.0%}"}
                elif before > 0 and drop > MERGE_DBCV_MAX_DROP:
                    critic_user = (
                        f"Fusion proposée (chute relative du DBCV : {drop:.0%}).\n"
                        f"Remédiation commune annoncée : {m['shared_remediation']}\n"
                        f"Justification : {m['justification']}\n\n"
                        + "\n\n".join(
                            _node_block(nodes[x], int((cur == x).sum()), titles, cards) for x in members
                        )
                    )
                    try:
                        cr = llm(SYSTEM_CRITIC_PROMPT, critic_user, cache_tag="merge_critic")
                        cr = cr if isinstance(cr, dict) else {}
                        component = str(cr.get("component") or "").strip()
                        action = str(cr.get("action") or "").strip()
                        # Les trois conditions sont exigées : un composant et une action nommés
                        ok = (
                            cr.get("single_specific_remediation") is True
                            and cr.get("addresses_all_members") is True
                            and bool(component)
                            and bool(action)
                        )
                        verdict = {
                            "single_specific_remediation": bool(ok),
                            "component": component,
                            "action": action,
                            "reason": str(cr.get("reason", "")),
                        }
                    except Exception:
                        ok, verdict = False, {"single_specific_remediation": False, "reason": "erreur du critique"}
                    vetoed = not ok
                flat = [f for x in members for f in nodes[x].members]
                # Un résultat évité rattaché à un nœud déjà fusionné est redistribué à ses clusters fins
                outcomes = [
                    {"fine_cluster_id": f, "outcome": o["outcome"]}
                    for o in m["prevented_outcomes"]
                    if o["fine_cluster_id"] in nodes
                    for f in nodes[o["fine_cluster_id"]].members
                ]
                entry = {
                    "round": rnd,
                    "action": "merge",
                    "members": flat,
                    "member_nodes": list(members),
                    "result_id": None if vetoed else new_id,
                    "shared_remediation": m["shared_remediation"],
                    "prevented_outcomes": outcomes,
                    "justification": m["justification"],
                    "dbcv_before": float(before),
                    "dbcv_after": float(after),
                    "vetoed": bool(vetoed),
                    "critic_verdict": verdict,
                }
                log.append(entry)
                if vetoed:
                    rejected.append(proposed)
                    continue
                cur = trial
                nodes[new_id] = _Node(new_id, flat)
                for x in members:
                    parent_of[x] = new_id
                    del nodes[x]
                accepted += 1

        # -------- drapeaux d'hétérogénéité (clusters fins non fusionnés uniquement)
        seen_flags: set[str] = set()
        for fl in pending_flags:
            fid = fl["fine_cluster_id"]
            if fid in seen_flags or fid not in nodes or nodes[fid].members != [fid] or fid not in set(fine.tolist()):
                continue
            seen_flags.add(fid)
            rows = np.where(cur == fid)[0]
            sizes = [s for s in MCS_SWEEP if s < len(rows) / 2]
            entry = {
                "round": rnd,
                "action": "flag_heterogeneous",
                "members": [fid],
                "result_id": None,
                "shared_remediation": "",
                "prevented_outcomes": [],
                "justification": fl["reason"],
                "dbcv_before": float(dbcv_score(X, cur)),
                "dbcv_after": None,
                "vetoed": False,
                "critic_verdict": None,
                "accepted": False,
            }
            # Pas de re-découpage récursif d'un sous-cluster déjà issu d'un découpage
            sw = sweep_hdbscan(X[rows], sizes) if sizes and "_s" not in fid else None
            sub = None if sw is None else np.asarray(sw.labels)
            sub_ids = [] if sub is None else sorted(set(sub.tolist()) - {-1})
            if 2 <= len(sub_ids) <= SPLIT_MAX_PARTS and sw.dbcv > 0:
                new_ids = [f"{fid}_s{j}" for j in range(len(sub_ids))]
                code = {s: new_ids[j] for j, s in enumerate(sub_ids)}
                sub_centroids = {s: X[rows][sub == s].mean(axis=0) for s in sub_ids}
                assigned = []
                for pos, s in enumerate(sub):
                    if s == -1:  # bruit : sous-cluster le plus proche dans l'espace UMAP
                        s = min(sub_ids, key=lambda q: float(np.linalg.norm(X[rows[pos]] - sub_centroids[q])))
                    assigned.append(code[s])
                trial_fine = fine.copy()
                trial_fine[rows] = np.array(assigned, dtype=object)
                new_cards, new_titles = {}, {}
                if relabel is not None:
                    try:
                        new_cards, new_titles = relabel(trial_fine.copy(), list(new_ids))
                    except Exception:
                        pass
                # Contrôle sémantique : si deux sous-groupes reçoivent le même intitulé, ce n'est
                # pas un mélange de modes de défaillance, le découpage est refusé
                sub_tokens = [_tokens(new_titles[n].get("fine_title", "")) for n in new_ids if n in new_titles]
                duplicate = any(
                    _jaccard(a, b) >= TITLE_JACCARD_MIN
                    for i, a in enumerate(sub_tokens)
                    for b in sub_tokens[i + 1:]
                )
                if duplicate:
                    entry["justification"] += " | découpage refusé : sous-groupes indiscernables"
                else:
                    fine[:] = trial_fine
                    cur[rows] = trial_fine[rows]
                    entry["dbcv_after"] = float(dbcv_score(X, cur))
                    entry["result_id"] = list(new_ids)
                    entry["accepted"] = True
                    parent_card, pt = cards.get(fid, {}), titles.get(fid, {})
                    for j, nid in enumerate(new_ids):
                        size = int((cur == nid).sum())
                        cards[nid] = dict(new_cards.get(nid) or _minimal_card(parent_card, size))
                        titles[nid] = dict(
                            new_titles.get(nid)
                            or {
                                "fine_title": f"{pt.get('fine_title', fid)} (sous-groupe {j})",
                                "failure_mode": pt.get("failure_mode", ""),
                            }
                        )
                        nodes[nid] = _Node(nid, [nid])
                    del nodes[fid]
                    accepted += 1
            log.append(entry)

        if accepted == 0:
            break

    # -------- numérotation finale par taille décroissante
    order = sorted(nodes.values(), key=lambda n: (-int((cur == n.id).sum()), sorted(n.members)[0]))
    final_id = {n.id: f"{partition}_M{k}" for k, n in enumerate(order)}
    fine_to_merged: dict[str, str] = {}
    merged_members: dict[str, list[str]] = {}
    for n in order:
        mid = final_id[n.id]
        merged_members[mid] = sorted(n.members)
        for f in n.members:
            fine_to_merged[f] = mid

    def resolve(x: str) -> str:
        while x in parent_of:
            x = parent_of[x]
        return final_id.get(x, x)

    for e in log:
        if e["action"] == "merge" and e["result_id"] is not None:
            e["result_id"] = resolve(e["result_id"])

    return MergeResult(
        fine_labels=fine,
        fine_to_merged=fine_to_merged,
        merged_members=merged_members,
        log=log,
        titles=titles,
    )
