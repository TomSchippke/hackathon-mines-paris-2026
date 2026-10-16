"""Test de bout en bout sur données synthétiques : cadrage, analyse, vérification, export HTML.

Tous les appels au LLM et les embeddings sont simulés : aucun appel Azure, aucun fichier .env lu. Le test
s'exécute dans un dossier temporaire (les chemins de la pipeline sont relatifs au dossier courant) et le
supprime à la fin.

Données : 2 segments × 3 sites, un mélange de causes différent par site, une cause contributive présente
sur un seul site (citation réelle), une citation contributive inventée (rejetée par le code), une cause
« inattendue » que l'affectation simulée range dans « aucun », et une conséquence de sévérité la plus faible
qui ressemble à un arrêt nominal.

Usage (depuis la racine du repo) :
    .venv/bin/python tests/e2e_mock.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.update(AZURE_OPENAI_CHAT_DEPLOYMENT="mock", AZURE_OPENAI_EMBEDDING_DEPLOYMENT="mock",
                  AZURE_OPENAI_API_KEY="x", AZURE_OPENAI_ENDPOINT="x", AZURE_OPENAI_API_VERSION="x")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- données synthétiques ---------------------------------------------------------------------
# cause : (phrase de cause, conséquence, sévérité, composant)
CAUSES = {
    "a": {
        "orage": ("un orage violent a provoqué une surtension sur les armoires électriques du poste",
                  "incendie partiel de l'armoire et destruction du matériel", "grave", "armoire 3"),
        "rongeurs": ("des rongeurs ont attaqué les gaines du câblage sous les rangées de modules",
                     "court-circuit sur le câble DC", "moyenne", "câble DC"),
        "usure": ("une usure avancée des roulements après des années de service intensif sans reprise",
                  "dégât matériel léger sur le roulement", "moyenne", "roulement"),
        "salissure": ("une accumulation de salissure sur les panneaux réduit nettement le rendement attendu",
                      "arrêt de sécurité automatique sans aucun dommage", "faible", "panneau 12"),
        "inattendu": ("une erreur de paramétrage du logiciel de supervision après la mise à jour nocturne",
                      "perte de supervision du site pendant plusieurs heures", "moyenne", "logiciel"),
    },
    "b": {
        "rafales": ("des rafales très fortes ont dépassé le seuil et déclenché la mise en sécurité de la pompe",
                    "arrêt de sécurité automatique sans dommage", "faible", "variateur"),
        "graissage": ("un défaut de graissage du palier principal non suivi par le plan d'entretien",
                      "dégât matériel sur le palier", "grave", "palier principal"),
        "fissure": ("une fissure du socle béton due au tassement du terrain meuble",
                    "fuite sur le socle", "moyenne", "socle"),
        "givre": ("du givre accumulé sur le rotor a déséquilibré la machine en hiver",
                  "perte d'activité de la pompe", "moyenne", "rotor"),
    },
}
SITES = {"a": ["Nord-1", "Nord-2", "Nord-3"], "b": ["Sud-1", "Sud-2", "Sud-3"]}
# Poids des causes par site : chaque site a sa dominante.
WEIGHTS = {"a": [[4, 2, 1, 1, 0.3], [1, 1, 1, 4, 0.3], [1, 1, 4, 1, 0.3]],
           "b": [[4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 1, 4]]}
CONTRIB = "le site est exposé à des vents salins permanents venant de la côte proche du parc"
INVENTED = "phrase totalement absente du rapport original ici"
UNKNOWN = "Origine non identifiée, arrêt constaté par le technicien de garde."
CONTEXT = "Le client exploite six sites de production : des postes de transformation (a) et des pompes (b)."
DOMAIN = "Le client exploite six sites de production : postes de transformation et pompes."


def head(sentence: str) -> str:
    """La phrase de cause telle qu'elle apparaît dans un rapport (majuscule au début)."""
    return sentence[0].upper() + sentence[1:]


def make_incidents(rng: random.Random) -> pd.DataFrame:
    rows, incident_id = [], 0
    for seg, sites in SITES.items():
        names = list(CAUSES[seg])
        for k, site in enumerate(sites):
            for _ in range(80):
                incident_id += 1
                cause = rng.choices(names, WEIGHTS[seg][k])[0]
                contrib = site == "Nord-3" and rng.random() < 0.7
                sentence = CAUSES[seg][cause][0] if rng.random() > 0.1 else None
                if sentence is None:  # causes non précisées, exclues de l'analyse des causes
                    body, cause = UNKNOWN, None
                else:
                    body = head(sentence) + "."
                text = (f"Rapport du technicien. {body} " + (f"Constat : {CONTRIB}. " if contrib else "")
                        + "Le technicien a sécurisé la zone et redémarré l'équipement après contrôle complet.")
                severity = CAUSES[seg][cause][2] if cause else rng.choice(["faible", "moyenne", "grave"])
                rows.append({"incident_id": incident_id, "site": site, "asset_type": seg,
                             "date": f"202{rng.randint(1, 3)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
                             "severity": severity, "downtime_hours": round(rng.uniform(1, 30), 1),
                             "estimated_cost_eur": round(rng.uniform(1000, 9000) * (4 if cause == "orage" else 1), 2),
                             "weather_condition": rng.choice(["pluie", "clair"]), "equipment_id": f"EQ{incident_id}",
                             "technician_id": f"T{k}", "text": text,
                             "_cause": cause, "_seg": seg})
    return pd.DataFrame(rows)


# --- LLM et embeddings simulés -----------------------------------------------------------------
def dummy(schema: dict):
    """Réponse conforme au schéma : premier choix des enums, listes vides, texte neutre sans chiffre."""
    kind = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object":
        return {k: dummy(v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "boolean":
        return False
    return "Texte simulé sans chiffre."


CALLS = {"cadrage": 0, "extraction": 0, "affectation": 0, "composants": 0, "redaction": 0, "jugement": 0, "agent": 0}
CADRAGE = {
    "domaine": DOMAIN,
    "segments": [{"valeur": "a", "libelle": "postes de transformation"}, {"valeur": "b", "libelle": "pompes"}],
    "severites": ["faible", "moyenne", "grave"],
}


def mock_chat(messages, *, role=None, tools=None, response_format=None, temperature=0.0):
    if tools:  # boucles d'agent : l'agent ne décide rien, le code prend le relais
        CALLS["agent"] += 1
        return {"content": "ok", "tool_calls": []}
    name = response_format["json_schema"]["name"]
    user = messages[-1]["content"]
    if name == "cadrage":
        CALLS["cadrage"] += 1
        assert CONTEXT in user and "DONNÉES" in user, "le cadrage doit recevoir le contexte et les données"
        return {"content": json.dumps(CADRAGE, ensure_ascii=False), "tool_calls": []}
    if name == "analyse_incident":
        CALLS["extraction"] += 1
        assert "Site" not in user.split("Description : ")[0] and "EQ" not in user, "site ou identifiant montré au LLM"
        text = user.split("Description : ", 1)[1]
        found = [(seg, c) for seg, causes in CAUSES.items() for c, (s, *_ ) in causes.items() if head(s) in text]
        out = dummy(response_format["json_schema"]["schema"])
        if not found:
            out.update(cause_type="inconnue", cause_racine="cause non précisée dans le rapport", consequence="arrêt constaté",
                       composant_brut="", confiance="faible")
        else:
            seg, cause = found[0]
            sentence, consequence, _, component = CAUSES[seg][cause]
            contrib = [{"texte": "exposition aux vents salins de la côte", "type": "explicite", "indice": CONTRIB[:70]}] \
                if CONTRIB in text else []
            contrib.append({"texte": "cause inventée", "type": "inférée", "indice": INVENTED})  # rejetée par le code
            # Une cause sur cinq est « inférée » (indice réel) : exerce l'échantillon de validation manuelle.
            inferred = int(hashlib.md5(text.encode()).hexdigest(), 16) % 5 == 0
            out.update(cause_racine=sentence, cause_type="inférée" if inferred else "explicite", indice=head(sentence)[:80],
                       causes_contributives=contrib, consequence=consequence, composant_brut=component,
                       confiance="haute", maintenance_suspectee="non")
        return {"content": json.dumps(out, ensure_ascii=False), "tool_calls": []}
    if name == "affectation":
        CALLS["affectation"] += 1
        themes = re.findall(r"\[(\S+)\] ([^:]+) :", user)
        phrases = re.findall(r"^(\d+)\. (.+)$", user.split("Phrases à classer")[1], re.M)
        affectations = []
        for index, phrase in phrases:
            if "paramétrage" in phrase:  # la cause inattendue : aucun thème ne convient
                theme = "aucun"
            else:
                theme = next((tid for tid, nm in themes if nm.strip().split()[0].lower()[:5] in phrase.lower()), "aucun")
            affectations.append({"index": int(index), "theme": theme})
        return {"content": json.dumps({"affectations": affectations}), "tool_calls": []}
    if name == "composants":
        CALLS["composants"] += 1
        values = re.findall(r"^- (.+)$", user, re.M)
        return {"content": json.dumps({"groupes": [{"nom": "composant", "valeurs": values}]}, ensure_ascii=False),
                "tool_calls": []}
    if name == "redaction":
        CALLS["redaction"] += 1
        m = re.search(r"Site : (\S+) \(.*?\)\. Cause : (.+?)\.\n", user)
        if m and "UNE mesure pour ce site" in user:
            site, cause = m.groups()
            word = re.findall(r"[^\W\d_]{5,}", cause.lower())[0]
            measure = {"type": "programme d'inspection",
                       "mesure": f"Sur {site}, inspecter chaque trimestre les équipements exposés au phénomène {word} "
                                 "et consigner les écarts",
                       "mecanisme": f"l'inspection repère l'exposition au phénomène {word} avant la panne"}
            return {"content": json.dumps(measure, ensure_ascii=False), "tool_calls": []}
        return {"content": json.dumps(dummy(response_format["json_schema"]["schema"]), ensure_ascii=False), "tool_calls": []}
    if name == "jugement":
        CALLS["jugement"] += 1
        return {"content": json.dumps(dummy(response_format["json_schema"]["schema"]), ensure_ascii=False), "tool_calls": []}
    raise ValueError(f"appel LLM non prévu : {name}")


def mock_embed(texts: list[str], cache_root: Path) -> np.ndarray:
    """Vecteurs : mots de 5 lettres et plus projetés sur 128 dimensions, plus un bruit fixe par phrase."""
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w{5,}", text.lower()):
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1
        out[i] += np.random.default_rng(i).normal(0, 0.05, 128)
    return out


def install_mocks() -> None:
    import analysis_agent.assignment as assignment
    import analysis_agent.components as components
    import analysis_agent.judge as judge
    import analysis_agent.writing as writing
    import cause_extraction.pipeline as extraction
    import framing.pipeline as framing
    import llm.agent as agent
    import topic_modeling.embeddings as embeddings

    for module in (framing, extraction, assignment, components, writing, judge, agent):
        module.chat = mock_chat
    embeddings.embed_texts = mock_embed


# --- scénario ----------------------------------------------------------------------------------
def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="e2e_mock_"))
    previous = Path.cwd()
    try:
        os.chdir(workdir)
        run_scenario(workdir)
    finally:
        os.chdir(previous)
        shutil.rmtree(workdir, ignore_errors=True)


def run_scenario(workdir: Path) -> None:
    Path("context").mkdir()
    Path("context/contexte.md").write_text(CONTEXT, encoding="utf-8")
    Path("outputs/1_extraction").mkdir(parents=True)
    incidents = make_incidents(random.Random(1))
    incidents.drop(columns=["_cause", "_seg"]).to_csv("outputs/1_extraction/incidents_full.csv", index=False)
    install_mocks()

    from framing.pipeline import run as run_framing
    from analysis_agent import facts as F
    from analysis_agent.pipeline import run as run_analysis
    from paths import ANALYSIS_DIR, REVIEW_DIR

    framing_result = run_framing()
    assert framing_result["severites"] == CADRAGE["severites"], "cadrage : severites inattendues"
    assert Path("outputs/1b_cadrage/cadrage.json").exists(), "cadrage.json absent"

    ok = run_analysis(max_turns=1)
    assert ok is True, "l'analyse n'a pas produit de rapport (voir outputs/2_analyse/journal.md)"

    report = (ANALYSIS_DIR / "rapport_analyse.md").read_text(encoding="utf-8")
    assert "Plan d'actions" in report, "rapport : section « Plan d'actions » absente"
    assert "Causes isolées" in report, "rapport : section « Causes isolées » absente"
    assert "profil" not in report.lower(), "rapport : vocabulaire de profils encore présent"

    decisions = json.loads((ANALYSIS_DIR / "decisions.json").read_text(encoding="utf-8"))
    assert "profils" not in decisions, "decisions.json : clé « profils » encore présente"
    rapport = decisions["rapport"]
    measures, coverage = rapport["mesures"], rapport["couverture"]
    assert len(measures) <= 12, f"plan : {len(measures)} mesures, 12 au plus"
    assert coverage["pct"] >= 80 or coverage["plafond_atteint"], f"plan : couverture {coverage['pct']} % sans plafond atteint"
    written = [m for m in measures if not m["a_definir"] and m["site"] in m["mesure"]]
    assert written, "plan : aucune mesure rédigée qui nomme son site"
    chains = decisions["chaines_prioritaires"]
    assert all(chains.get(seg, {}).get("chaines") for seg in ("a", "b")), f"chaînes prioritaires absentes : {chains}"
    assert Path("outputs/validation/inferences_a_valider.csv").exists(), "échantillon des causes inférées absent"
    assert decisions["isoles"]["a"]["phrases"] > 0 and decisions["isoles"]["a"]["exemples"], \
        "causes isolées : la cause inattendue n'apparaît pas comme isolée"

    incidents_csv = pd.read_csv(ANALYSIS_DIR / "incidents_analyse.csv")
    rejected = int(incidents_csv["contributives_rejetees"].sum())
    assert rejected > 0, "aucune citation contributive inventée rejetée par le code"

    phrases = pd.read_csv(ANALYSIS_DIR / "phrases.csv")
    contrib = phrases[(phrases["role"] == "contributive") & phrases["texte"].str.contains("vents salins", na=False)]
    assert len(contrib) and contrib["theme_id"].notna().all(), "la cause contributive n'est rangée dans aucun thème"
    contributive_only = {t for ids in incidents_csv["themes_contributifs_ids"].dropna() for t in ids.split("|")} \
        - set(incidents_csv["theme_cause_id"].dropna())
    assert contributive_only, "aucun thème n'existe seulement comme cause contributive"

    import webapp.export_html as export
    figures = sys.modules["figures"]
    figures.ANALYSIS = Path("outputs/2_analyse").resolve()
    figures.REVIEW = Path("outputs/3_verification").resolve()
    export.OUTPUT = figures.ANALYSIS / "rapport_interactif.html"
    names = figures.theme_names()
    site_criteria = F.site_indicators(incidents_csv[incidents_csv["asset_type"] == "a"], names)
    assert any(names.get(t) in site_criteria for t in contributive_only), \
        "le thème contributif n'apparaît dans aucun critère de site"

    # La revue est lancée ensuite : son échec (réécriture de review_agent en cours) ne bloque pas le reste.
    review_error = None
    try:
        from review_agent.pipeline import run as run_review
        run_review(max_turns=1)
    except Exception as e:  # noqa: BLE001 - rapporté à la fin du test
        review_error = e
        print(f"Revue : échec ({type(e).__name__}: {e})")

    html_path = export.build()
    assert html_path.exists(), "rapport_interactif.html absent"
    page = html_path.read_text(encoding="utf-8")
    assert "Plan" in page and "Causes isolées" in page, "export : sections du plan ou des causes isolées absentes"

    print(f"Analyse : {len(measures)} mesures, couverture {coverage['pct']} % (plafond : {coverage['plafond_atteint']}), "
          f"{rejected} citations contributives rejetées, {len(contributive_only)} thème(s) contributif(s) seul(s)")
    print(f"Appels simulés : {CALLS}")
    if review_error is not None:
        raise AssertionError(f"étape revue en échec (review_agent en cours de réécriture ?) : {review_error!r}")
    assert (REVIEW_DIR / "verification.md").exists(), "verification.md absent"
    print("E2E OK")


if __name__ == "__main__":
    main()
