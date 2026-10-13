"""Les outils de l'agent vérificateur. Lecture seule : il ne modifie rien de l'analyse.

    automatic_checks   contrôles faits par le code (donnés à l'agent dès le départ)
    read_report        le rapport d'analyse
    read_journal       le journal de l'agent d'analyse, par pages
    site_facts         chiffres recalculés d'un site, à côté de ce que le rapport en dit
    theme_facts        contenu réel d'un thème (sites, facteurs, exemples)
    record_finding     consigne un problème
    conclude           verdict final (outil final)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_agent.config import (ACTIONS_PER_SITE, ASSET_TYPES, DIMENSIONS, MAX_CLUSTERING_RUNS,
                                   MAX_UNCLASSIFIED_PCT, MIN_STABILITY, N_TOPICS_RANGE, SITE_PROFILES)
from cause_extraction.config import CAUSE_INCONNUE
from analysis_agent.tools import percent
from llm.agent import ToolError

from .config import GRAVITES, JOURNAL_PAGE_MAX, RESULT_MAX_CHARS, VERDICTS


def truncate(value, limit: int = RESULT_MAX_CHARS):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= limit else text[:limit] + " [...]"


class ReviewWorkspace:
    def __init__(self, analysis_dir: Path):
        self.dir = analysis_dir
        files = {"journal": "journal.json", "decisions": "decisions.json", "incidents": "incidents_analyse.csv"}
        missing = [f for f in files.values() if not (analysis_dir / f).exists()]
        if missing:
            raise SystemExit(f"Fichiers absents dans {analysis_dir} : {missing}. Lancer d'abord l'étape analyse.")
        self.journal = json.loads((analysis_dir / "journal.json").read_text(encoding="utf-8"))
        self.decisions = json.loads((analysis_dir / "decisions.json").read_text(encoding="utf-8"))
        self.incidents = pd.read_csv(analysis_dir / "incidents_analyse.csv")
        if "erreur_analyse" in self.incidents:  # une chaîne vide est relue comme NaN par pandas
            self.incidents["erreur_analyse"] = self.incidents["erreur_analyse"].fillna("")
        report = analysis_dir / "rapport_analyse.md"
        self.report_md = report.read_text(encoding="utf-8") if report.exists() else None
        self.report = self.decisions.get("rapport")
        self.findings: list[dict] = []
        self.verdict: dict | None = None
        self.report_read = False
        self.checks = self.automatic_checks()

    @property
    def tools(self) -> dict:
        return {"read_report": self.read_report, "read_journal": self.read_journal, "site_facts": self.site_facts,
                "theme_facts": self.theme_facts, "record_finding": self.record_finding, "conclude": self.conclude}

    # ---- contrôles automatiques -------------------------------------------------------------------

    def automatic_checks(self) -> list[dict]:
        checks = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append({"controle": name, "ok": bool(ok), "detail": detail})

        check("rapport produit", self.report is not None and self.report_md is not None,
              "rapport_analyse.md présent" if self.report_md else "l'agent n'a pas terminé : pas de rapport")
        essais = self.decisions.get("essais", {})
        chosen = self.decisions.get("clustering_choisi", {})
        for asset_type in ASSET_TYPES:
            sub = self.incidents[self.incidents["asset_type"] == asset_type]
            analysed = sub["erreur_analyse"].eq("") if "erreur_analyse" in sub else pd.Series(False, index=sub.index)
            check(f"{asset_type} : causes et conséquences extraites", analysed.mean() >= 0.95,
                  f"{analysed.sum()}/{len(sub)} incidents analysés par le LLM")
            if "cause_racine" in sub:
                unknown = 100 * (sub["cause_racine"] == CAUSE_INCONNUE).mean()
                check(f"{asset_type} : causes non précisées dans les rapports (information)", True,
                      f"{unknown:.0f} % des rapports ne donnent pas la cause : limite des données, à mentionner "
                      "dans le rapport")
            for d in DIMENSIONS:
                k = f"{asset_type}/{d}"
                n_runs = sum(e.get("asset_type") == asset_type and e.get("dimension") == d for e in essais.values())
                check(f"{k} : budget de clustering", n_runs <= MAX_CLUSTERING_RUNS,
                      f"{n_runs} essais (maximum {MAX_CLUSTERING_RUNS})")
                if k not in chosen:
                    check(f"{k} : clustering choisi", False, "aucun clustering choisi")
                    continue
                e = essais[chosen[k]["run_id"]]
                check(f"{k} : qualité du clustering choisi",
                      e["stabilite"] >= MIN_STABILITY and e["non_classes_pct"] <= MAX_UNCLASSIFIED_PCT
                      and N_TOPICS_RANGE[0] <= e["nombre_clusters"] <= N_TOPICS_RANGE[1],
                      f"{chosen[k]['run_id']} : stabilité {e['stabilite']}, non classés {e['non_classes_pct']} %, "
                      f"{e['nombre_clusters']} clusters, DBCV {e['dbcv']}")
                classified = sub[sub[f"cluster_{d}"].notna() & (sub[f"cluster_{d}"] != -1)]
                check(f"{k} : chaque cluster dans un thème", classified[f"theme_{d}_id"].notna().all(),
                      f"{classified[f'theme_{d}_id'].isna().sum()} incidents classés sans thème")
        if self.report:
            for asset_type in ASSET_TYPES:
                sites = set(self.incidents.loc[self.incidents["asset_type"] == asset_type, "site"])
                profiles = sorted(s["profil"] for s in self.report["sites"] if s["site"] in sites)
                check(f"{asset_type} : profils A, B, C attribués une fois chacun",
                      profiles == sorted(SITE_PROFILES[asset_type]), f"profils attribués : {profiles}")
            counts = {s["site"]: len(s["plan_remediation"]) for s in self.report["sites"]}
            check("plan de remédiation pour chaque site",
                  all(ACTIONS_PER_SITE[0] <= n <= ACTIONS_PER_SITE[1] for n in counts.values()), str(counts))
        calls = [e for e in self.journal if e["type"] == "outil"]
        refused = [e for e in calls if not e["resultat"].get("ok")]
        check("refus d'outils dans le journal (information)", True,
              f"{len(refused)} refus sur {len(calls)} appels : "
              + ", ".join(sorted({e['outil'] for e in refused})) if refused else f"0 refus sur {len(calls)} appels")
        return checks

    # ---- outils -----------------------------------------------------------------------------------

    def read_report(self) -> dict:
        if self.report_md is None:
            raise ToolError("pas de rapport : l'agent d'analyse n'a pas terminé")
        self.report_read = True
        return {"rapport": self.report_md}

    def read_journal(self, start: int, count: int) -> dict:
        if not isinstance(start, int) or not 0 <= start < len(self.journal):
            raise ToolError(f"start doit être entre 0 et {len(self.journal) - 1}")
        if not isinstance(count, int) or not 1 <= count <= JOURNAL_PAGE_MAX:
            raise ToolError(f"count doit être entre 1 et {JOURNAL_PAGE_MAX}")
        events = [{**e, "resultat": truncate(e["resultat"])} if e["type"] == "outil" else e
                  for e in self.journal[start:start + count]]
        return {"evenements": events, "total": len(self.journal), "suivant": min(start + count, len(self.journal))}

    def site_facts(self, site: str) -> dict:
        sites = sorted(self.incidents["site"].unique())
        if site not in sites:
            raise ToolError(f"site inconnu : {site!r}. Sites : {sites}")
        g = self.incidents[self.incidents["site"] == site]
        ok = g[g["erreur_analyse"] == ""] if "erreur_analyse" in g else g
        facts = {
            "incidents": len(g),
            "themes_causes_pct": percent(g["theme_cause"].fillna("non classé"), top=8),
            "themes_consequences_pct": percent(g["theme_consequence"].fillna("non classé"), top=6),
            "paires_cause_consequence": {
                f"{cause} -> {consequence}": int(n) for (cause, consequence), n in
                g.groupby([g["theme_cause"].fillna("non classé"), g["theme_consequence"].fillna("non classé")])
                .size().sort_values(ascending=False).head(5).items()},
            "facteurs_pct": percent(ok["facteur_principal"], top=8) if "facteur_principal" in ok else None,
            "lie_vieillissement_pct": round(100 * ok["lie_vieillissement"].astype(bool).mean(), 1)
            if "lie_vieillissement" in ok else None,
            "lie_maintenance_pct": round(100 * ok["lie_maintenance"].astype(bool).mean(), 1)
            if "lie_maintenance" in ok else None,
            "severe_pct": round(100 * (g["severity"] == "sévère").mean(), 1),
            "cout_moyen_eur": round(g["estimated_cost_eur"].mean()), "arret_moyen_h": round(g["downtime_hours"].mean(), 1),
            "meteo_pct": percent(g["weather_condition"], top=5),
        }
        claimed = next((s for s in (self.report or {}).get("sites", []) if s["site"] == site), None)
        asset_type = g["asset_type"].iloc[0]
        return {"site": site, "chiffres_recalcules": facts, "profils_possibles": SITE_PROFILES[asset_type],
                "ce_que_dit_le_rapport": claimed}

    def theme_facts(self, theme_id: str) -> dict:
        for d, column in DIMENSIONS.items():
            g = self.incidents[self.incidents[f"theme_{d}_id"] == theme_id]
            if len(g):
                sample = g.sample(min(4, len(g)), random_state=0)
                return {"theme_id": theme_id, "dimension": d, "nom": g[f"theme_{d}"].iloc[0], "incidents": len(g),
                        "sites_pct": percent(g["site"]), "facteurs_pct": percent(g["facteur_principal"], top=5),
                        f"{column}_exemples": sample[column].tolist(),
                        "description_exemple": [t[:300] for t in sample["text"].head(1)]}
        themes = sorted(set(self.incidents["theme_cause_id"].dropna()) | set(self.incidents["theme_consequence_id"].dropna()))
        raise ToolError(f"thème inconnu : {theme_id!r}. Thèmes : {themes}")

    def record_finding(self, gravite: str, element: str, constat: str, preuve: str) -> dict:
        if gravite not in GRAVITES:
            raise ToolError(f"gravite doit valoir : {GRAVITES}")
        for name, value in {"element": element, "constat": constat, "preuve": preuve}.items():
            if not isinstance(value, str) or not value.strip():
                raise ToolError(f"{name} ne doit pas être vide")
        self.findings.append({"gravite": gravite, "element": element.strip(), "constat": constat.strip(),
                              "preuve": preuve.strip()})
        return {"constats_enregistres": len(self.findings)}

    def conclude(self, verdict: str, synthese: str) -> dict:
        if verdict not in VERDICTS:
            raise ToolError(f"verdict doit valoir : {VERDICTS}")
        if self.report_md is not None and not self.report_read:
            raise ToolError("lire le rapport (read_report) avant de conclure")
        failed = [c["controle"] for c in self.checks if not c["ok"]]
        blocking = [f for f in self.findings if f["gravite"] == "bloquant"]
        major = [f for f in self.findings if f["gravite"] == "majeur"]
        if verdict == "valide" and (failed or blocking or major):
            raise ToolError(f"verdict 'valide' impossible : {len(failed)} contrôles automatiques en échec, "
                            f"{len(blocking)} constats bloquants, {len(major)} constats majeurs. Avec des constats "
                            "majeurs : 'valide_avec_reserves' au mieux ; avec un constat bloquant : 'a_refaire'.")
        if verdict == "valide_avec_reserves" and (failed or blocking):
            raise ToolError(f"verdict 'valide_avec_reserves' impossible : {len(failed)} contrôles en échec, "
                            f"{len(blocking)} constats bloquants -> 'a_refaire'")
        if not isinstance(synthese, str) or not synthese.strip():
            raise ToolError("synthese ne doit pas être vide")
        self.verdict = {"verdict": verdict, "synthese": synthese.strip()}
        return {"message": "verdict enregistré"}


TOOLS = [
    {"type": "function", "function": {
        "name": "read_report", "description": "Le rapport d'analyse complet (Markdown).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_journal", "description": f"Événements du journal de l'agent d'analyse, à partir de `start` "
                                              f"({JOURNAL_PAGE_MAX} maximum par appel).",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "integer", "minimum": 0},
            "count": {"type": "integer", "minimum": 1, "maximum": JOURNAL_PAGE_MAX}},
            "required": ["start", "count"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "site_facts", "description": "Chiffres recalculés d'un site à partir des données, à côté de ce que "
                                            "le rapport affirme pour ce site.",
        "parameters": {"type": "object", "properties": {"site": {"type": "string"}},
                       "required": ["site"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "theme_facts", "description": "Contenu réel d'un thème de causes ou de conséquences : taille, sites, "
                                             "facteurs, phrases extraites, exemple de description.",
        "parameters": {"type": "object", "properties": {"theme_id": {"type": "string"}},
                       "required": ["theme_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "record_finding", "description": "Consigne un point vérifié : 'conforme' s'il est correct, sinon "
                                                "un problème (bloquant, majeur, mineur), avec sa preuve.",
        "parameters": {"type": "object", "properties": {
            "gravite": {"type": "string", "enum": GRAVITES},
            "element": {"type": "string", "description": "partie concernée (site, thème, clustering...)"},
            "constat": {"type": "string"}, "preuve": {"type": "string", "description": "chiffres ou extrait"}},
            "required": ["gravite", "element", "constat", "preuve"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "conclude", "description": "Outil final : verdict et synthèse de la vérification.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": VERDICTS}, "synthese": {"type": "string"}},
            "required": ["verdict", "synthese"], "additionalProperties": False}}},
]
