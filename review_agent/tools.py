"""Les outils de l'agent vérificateur. Lecture seule : il ne modifie rien de l'analyse.

    automatic_checks   contrôles faits par le code (donnés à l'agent dès le départ)
    read_report        le rapport d'analyse
    read_journal       le journal de l'agent d'analyse, par pages
    site_facts         chiffres recalculés d'un site, à côté de ce que le rapport en dit
    theme_facts        contenu réel d'un thème (phrases, sites, exemples)
    record_finding     consigne un point vérifié ou un problème
    conclude           verdict final (outil final)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_agent import facts as F
from analysis_agent.config import (CONSEQUENCE_WORDS, DIMENSIONS, MAX_CLUSTERING_RUNS, MAX_UNCLASSIFIED_PCT,
                                   MIN_STABILITY, N_TOPICS_RANGE, NOMINAL_MIN_MILDEST_PCT, NONE_MAX_PCT,
                                   PLAN_COVERAGE_PCT, PLAN_MAX_MEASURES)
from analysis_agent.tools import percent
from analysis_agent.writing import measure_errors
from cause_extraction.config import CAUSE_INCONNUE
from framing.cadrage import current
from llm.agent import ToolError

from .config import FIELDS, GRAVITES, JOURNAL_PAGE_MAX, RESULT_MAX_CHARS, TARGETS_EXTRA, VERDICTS


# Même motif que CONSEQUENCE_WORDS, sans groupes de capture (pandas le signale sinon).
NON_CAPTURING_CONSEQUENCE = CONSEQUENCE_WORDS.replace("(", "(?:")


def truncate(value, limit: int = RESULT_MAX_CHARS):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= limit else text[:limit] + " [...]"


def read_csv_if(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


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
        self.themes = read_csv_if(analysis_dir / "themes.csv")
        self.phrases = read_csv_if(analysis_dir / "phrases.csv")
        # Nom de chaque thème de cause (par identifiant) : les critères des sites se lisent avec ces noms.
        causes = self.themes[self.themes["dimension"] == "cause"] if self.themes is not None else pd.DataFrame()
        self.theme_names = dict(zip(causes.get("theme_id", []), causes.get("name", [])))
        self._indicators: dict[str, dict] = {}
        self.findings: list[dict] = []
        self.verdict: dict | None = None
        self.report_read = False
        self.sites = sorted(self.incidents["site"].unique())
        self.cadrage = current()
        self.context_columns = [c for c in self.decisions.get("colonnes_contexte", []) if c in self.incidents]
        self.checks = self.automatic_checks()

    @property
    def tools(self) -> dict:
        return {"read_report": self.read_report, "read_journal": self.read_journal, "site_facts": self.site_facts,
                "theme_facts": self.theme_facts, "record_finding": self.record_finding, "conclude": self.conclude}

    def indicators(self, asset_type: str) -> dict[str, dict]:
        """Critères mesurés sur les sites d'un segment (facts.site_indicators), avec les noms des thèmes."""
        if asset_type not in self._indicators:
            segment = self.incidents[self.incidents["asset_type"] == asset_type]
            self._indicators[asset_type] = F.site_indicators(segment, self.theme_names)
        return self._indicators[asset_type]

    # ---- contrôles automatiques -------------------------------------------------------------------

    def automatic_checks(self) -> list[dict]:
        checks = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append({"controle": name, "ok": bool(ok), "detail": detail})

        check("rapport produit", self.report is not None and self.report_md is not None,
              "rapport_analyse.md présent" if self.report_md else "l'agent n'a pas terminé : pas de rapport")
        essais = self.decisions.get("essais", {})
        chosen = self.decisions.get("clustering_choisi", {})
        isoles = self.decisions.get("isoles", {})
        nominaux = self.decisions.get("nominaux", {})
        for asset_type in self.cadrage.segments:
            sub = self.incidents[self.incidents["asset_type"] == asset_type]
            analysed = sub["erreur_analyse"].eq("") if "erreur_analyse" in sub else pd.Series(False, index=sub.index)
            check(f"{asset_type} : causes et conséquences extraites", analysed.mean() >= 0.95,
                  f"{analysed.sum()}/{len(sub)} incidents analysés par le LLM")
            rejected = int(sub["contributives_rejetees"].sum()) if "contributives_rejetees" in sub else 0
            check(f"{asset_type} : causes contributives écartées faute de preuve (information)", True,
                  f"{rejected} causes contributives écartées par le code (citation absente du texte)")
            if "cause_racine" in sub:
                unknown = 100 * (sub["cause_racine"] == CAUSE_INCONNUE).mean()
                check(f"{asset_type} : causes non précisées dans les rapports (information)", True,
                      f"{unknown:.0f} % des rapports ne donnent pas la cause : limite des données, à mentionner "
                      "dans le rapport")
            for d in DIMENSIONS:
                k = f"{asset_type}/{d}"
                # Les essais de reprise (thèmes ajoutés aux phrases non rangées) ne comptent pas dans le budget.
                runs = [e for e in essais.values() if e.get("asset_type") == asset_type
                        and e.get("dimension") == d and not e.get("reprise")]
                check(f"{k} : budget de clustering", len(runs) <= MAX_CLUSTERING_RUNS,
                      f"{len(runs)} essai(s) (maximum {MAX_CLUSTERING_RUNS})")
                run_id = chosen.get(k, {}).get("run_id")
                e = essais.get(run_id) if run_id else None
                if not e:
                    check(f"{k} : clustering choisi", False, "aucun clustering choisi")
                elif not e.get("reprise"):
                    check(f"{k} : qualité du clustering choisi",
                          e["stabilite"] >= MIN_STABILITY and e["non_classes_pct"] <= MAX_UNCLASSIFIED_PCT
                          and N_TOPICS_RANGE[0] <= e["nombre_clusters"] <= N_TOPICS_RANGE[1],
                          f"{run_id} : stabilité {e['stabilite']}, non classés {e['non_classes_pct']} %, "
                          f"{e['nombre_clusters']} clusters, DBCV {e['dbcv']}")
                if self.phrases is not None:
                    ph = self.phrases[(self.phrases["asset_type"] == asset_type) & (self.phrases["dimension"] == d)]
                    assigned = int(ph["theme_id"].notna().sum())
                    check(f"{k} : phrases rattachées à un thème (information)", True,
                          f"{assigned}/{len(ph)} phrases ({F.share(assigned, len(ph))} %) ; le reste est compté "
                          "dans les causes isolées")
            iso = isoles.get(asset_type, {})
            pct = iso.get("pct", 0.0)
            check(f"{asset_type} : causes isolées, non regroupées (information)", True,
                  f"{pct} % des phrases de cause ({iso.get('phrases', 0)} phrases), seuil {NONE_MAX_PCT} %"
                  + (" : dépassé, constat mineur" if pct > NONE_MAX_PCT else ""))
            if pct > NONE_MAX_PCT:
                self.auto_finding("mineur", "general", "autre",
                                  f"{pct} % des phrases de cause du segment {asset_type} restent sans thème "
                                  f"(seuil {NONE_MAX_PCT} %)", "exemples : " + " | ".join(iso.get("exemples", [])),
                                  "Indique dans le rapport que ces causes isolées ne sont pas regroupées en thème, "
                                  "et ne les présente pas comme une cause dominante.")
            check_nominal = self.nominal_check(asset_type, nominaux.get(asset_type, {"theme_ids": [], "justification": ""}))
            checks.append(check_nominal)

        if self.report:
            self.plan_checks(check)
            self.report_checks(check)
        themes = self.themes
        mixed = []
        if themes is not None:
            mixed = themes.loc[(themes["dimension"] == "cause") & themes["name"].str.contains(
                NON_CAPTURING_CONSEQUENCE, case=False, regex=True), "name"].tolist()
        check("noms des thèmes de causes sans conséquence", not mixed, f"à renommer : {mixed}" if mixed else "ok")
        auto = [e["texte"] for e in self.journal if e["type"] == "automatique"]
        check("reprises automatiques par le code (information)", True,
              f"{len(auto)} : " + " | ".join(auto) if auto else "aucune : l'agent et le rédacteur ont tout fait")
        calls = [e for e in self.journal if e["type"] == "outil"]
        refused = [e for e in calls if not e["resultat"].get("ok")]
        check("refus d'outils dans le journal (information)", True,
              f"{len(refused)} refus sur {len(calls)} appels : "
              + ", ".join(sorted({e['outil'] for e in refused})) if refused else f"0 refus sur {len(calls)} appels")
        return checks

    def nominal_check(self, asset_type: str, nominal: dict) -> dict:
        """Thèmes de conséquences choisis comme arrêts nominaux : la part de leurs incidents à la sévérité la plus
        faible doit atteindre NOMINAL_MIN_MILDEST_PCT (recalculée ici, pas reprise de l'agent)."""
        sub = self.incidents[self.incidents["asset_type"] == asset_type]
        mildest = self.cadrage.mildest
        parts = []
        for theme_id in nominal.get("theme_ids", []):
            g = sub[sub["theme_consequence_id"] == theme_id]
            pct = F.share(int((g["severity"] == mildest).sum()), len(g))
            name = g["theme_consequence"].iloc[0] if len(g) else theme_id
            parts.append((name, pct, len(g)))
        justification = nominal.get("justification", "")
        automatic = justification.startswith(("CHOIX AUTOMATIQUE", "AUCUN THÈME NOMINAL (choix automatique"))
        origin = "choix automatique par le code" if automatic else "choisis par l'agent"
        if not parts:
            detail = f"aucun thème nominal ; {justification}"
        else:
            detail = (f"{origin} : " + " | ".join(f"{n} ({p} % à la sévérité « {mildest} », {k} incidents)"
                                                 for n, p, k in parts) + f" ; {justification}")
        ok = all(p >= NOMINAL_MIN_MILDEST_PCT for _, p, _ in parts)
        return {"controle": f"{asset_type} : arrêts nominaux (seuil {NOMINAL_MIN_MILDEST_PCT:g} %)", "ok": ok,
                "detail": detail if ok else f"sous le seuil : {detail}"}

    def plan_checks(self, check) -> None:
        """Couverture du plan, nombre de mesures, et chaque mesure (définie, sur son site, préventive)."""
        cov = self.report.get("couverture", {})
        pct, capped = cov.get("pct", 0.0), bool(cov.get("plafond_atteint"))
        mesures = self.report.get("mesures", [])
        ok = pct >= PLAN_COVERAGE_PCT or capped
        detail = f"{pct} % du coût des causes connues couvert par {len(mesures)} mesures"
        if capped and pct < PLAN_COVERAGE_PCT:
            detail += f" ; plafond de {PLAN_MAX_MEASURES} mesures atteint : à signaler dans le rapport"
        check(f"plan : couverture du coût (seuil {PLAN_COVERAGE_PCT:g} %)", ok, detail)
        check(f"plan : nombre de mesures (maximum {PLAN_MAX_MEASURES})", len(mesures) <= PLAN_MAX_MEASURES,
              f"{len(mesures)} mesures")
        to_define, to_revise = 0, 0
        for m in mesures:
            figures = (f"{m['incidents']} incidents ; {F.euros(m['cout_par_an'])} par an ; {m['part_cout_pct']} % "
                       "du coût des causes connues")
            if m.get("a_definir"):
                to_define += 1
                self.auto_finding("majeur", m["site"], "mesures", f"mesure à définir pour « {m['cause']} » sur "
                                  f"{m['site']} : aucune action proposée", figures,
                                  f"Propose une action de terrain préventive sur {m['site']} pour « {m['cause']} » "
                                  f"(inspection, protection, nettoyage planifié...) avec son mécanisme, en t'appuyant "
                                  f"sur les chiffres : {figures}.")
                continue
            others = [s for s in self.sites if s != m["site"]]
            errors = measure_errors(m, m["cause"], m["site"], others)
            if errors:
                to_revise += 1
                self.auto_finding("majeur", m["site"], "mesures", f"mesure {m['id']} à revoir pour « {m['cause']} » : "
                                  f"{m['mesure']}", "; ".join(errors), f"Réécris la mesure {m['id']} : "
                                  + "; ".join(errors) + f". Chiffres : {figures}.")
        check("mesures : définies, sur leur site, préventives (information)", True,
              f"{len(mesures)} mesures : {to_define} à définir, {to_revise} à revoir (constats majeurs)")

    def report_checks(self, check) -> None:
        """Contrôles du rapport : chaînes prioritaires, origine des causes dominantes."""
        chains = self.decisions.get("chaines_prioritaires", {})
        for asset_type in self.cadrage.segments:
            choice = chains.get(asset_type)
            if not choice or not choice.get("chaines"):
                check(f"{asset_type} : chaînes cause -> conséquence prioritaires (heatmap)", False,
                      "aucune chaîne choisie")
                continue
            origin = ("choix automatique par le code" if choice["justification"].startswith("CHOIX AUTOMATIQUE")
                      else "choisies par l'agent")
            pairs = ", ".join(f"{c['cause']} -> {c['consequence']}" for c in choice["chaines"])
            check(f"{asset_type} : chaînes cause -> conséquence prioritaires (heatmap)", True, f"{origin} : {pairs}")
        for s in self.report["sites"]:
            origin = s.get("origine_causes_dominantes", {})
            if origin.get("inférée", 0) > 50:
                self.auto_finding("mineur", s["site"], "causes_dominantes",
                                  f"les causes dominantes reposent surtout sur des inférences ({origin['inférée']} %)",
                                  str(origin), "Précise que ces causes sont majoritairement inférées.")

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
        asset_type = g["asset_type"].iloc[0]
        indicators = self.indicators(asset_type)
        ranking = F.rank_text(indicators)
        criteria = {c: {"valeur": d["valeurs"][site], "unite": d["unite"], "classement": ranking[c]}
                    for c, d in indicators.items() if site in d["valeurs"]}
        facts = {
            "incidents": len(g),
            "themes_causes_pct": percent(g["theme_cause"].fillna("non classé"), top=8),
            "themes_consequences_pct": percent(g["theme_consequence"].fillna("non classé"), top=6),
            "paires_cause_consequence": {
                f"{cause} -> {consequence}": int(n) for (cause, consequence), n in
                g.groupby([g["theme_cause"].fillna("non classé"), g["theme_consequence"].fillna("non classé")])
                .size().sort_values(ascending=False).head(5).items()},
            "criteres_du_segment": criteria,
            "arret_moyen_h": round(g["downtime_hours"].mean(), 1),
            **{f"{c}_pct": percent(g[c], top=5) for c in self.context_columns},
        }
        claimed = next((s for s in (self.report or {}).get("sites", []) if s["site"] == site), None)
        if claimed is not None:
            measures = {m["id"]: m for m in (self.report or {}).get("mesures", [])}
            claimed = {**claimed, "mesures": [
                {k: measures[i][k] for k in ("id", "cause", "mesure", "mecanisme", "a_definir") if k in measures[i]}
                for i in claimed.get("mesures", []) if i in measures]}
        return {"site": site, "chiffres_recalcules": facts, "ce_que_dit_le_rapport": claimed}

    def theme_facts(self, theme_id: str) -> dict:
        if self.phrases is not None and theme_id in set(self.phrases["theme_id"].dropna()):
            return self.theme_from_phrases(theme_id)
        return self.theme_from_incidents(theme_id)

    def theme_from_phrases(self, theme_id: str) -> dict:
        """Thème décrit par ses phrases (phrases.csv) : nombre, rôles, sites, exemples."""
        ph = self.phrases[self.phrases["theme_id"] == theme_id]
        # position = numéro de ligne de l'incident dans incidents_analyse.csv (écrit dans l'ordre de l'analyse),
        # pas incident_id : les incidents sans texte sont retirés avant l'analyse.
        site_of = dict(enumerate(self.incidents["site"].tolist()))
        sites = ph["position"].map(site_of).dropna()
        meta = self.themes[self.themes["theme_id"] == theme_id] if self.themes is not None else pd.DataFrame()
        description = str(meta["description"].iloc[0])[:300] if len(meta) and "description" in meta else ""
        examples = ph["texte"].drop_duplicates().sample(min(4, ph["texte"].nunique()), random_state=0)
        return {"theme_id": theme_id, "dimension": ph["dimension"].iloc[0],
                "nom": meta["name"].iloc[0] if len(meta) else theme_id, "description": description,
                "phrases": len(ph), "roles": ph["role"].value_counts().to_dict(),
                "incidents": int(ph["position"].nunique()), "sites_pct": percent(sites),
                "exemples": [t[:300] for t in examples.tolist()]}

    def theme_from_incidents(self, theme_id: str) -> dict:
        """Repli sans phrases.csv : le thème vu dans la table des incidents (cause ou conséquence)."""
        for id_col, name_col, text_col, dimension in (("theme_cause_id", "theme_cause", "cause_racine", "cause"),
                                                      ("theme_consequence_id", "theme_consequence", "consequence",
                                                       "consequence")):
            g = self.incidents[self.incidents[id_col] == theme_id]
            if len(g):
                sample = g.sample(min(4, len(g)), random_state=0)
                return {"theme_id": theme_id, "dimension": dimension, "nom": g[name_col].iloc[0], "incidents": len(g),
                        "sites_pct": percent(g["site"]),
                        "exemples": [t[:300] for t in sample[text_col].tolist()]}
        themes = sorted(set(self.incidents["theme_cause_id"].dropna()) | set(self.incidents["theme_consequence_id"].dropna()))
        raise ToolError(f"thème inconnu : {theme_id!r}. Thèmes : {themes}")

    def auto_finding(self, gravite: str, site: str, champ: str, constat: str, preuve: str, consigne: str) -> None:
        """Constat produit par un contrôle déterministe (pas par l'agent)."""
        self.findings.append({"gravite": gravite, "cible_site": site, "cible_champ": champ, "constat": constat,
                              "preuve": preuve, "consigne": consigne, "source": "code"})

    def record_finding(self, gravite: str, cible_site: str, cible_champ: str, constat: str, preuve: str,
                       consigne: str = "") -> dict:
        if gravite not in GRAVITES:
            raise ToolError(f"gravite doit valoir : {GRAVITES}")
        if cible_site not in self.sites + TARGETS_EXTRA:
            raise ToolError(f"cible_site doit valoir l'un de : {self.sites + TARGETS_EXTRA}")
        if cible_champ not in FIELDS:
            raise ToolError(f"cible_champ doit valoir l'un de : {FIELDS}")
        for name, value in {"constat": constat, "preuve": preuve}.items():
            if not isinstance(value, str) or not value.strip():
                raise ToolError(f"{name} ne doit pas être vide")
        if gravite in ("bloquant", "majeur") and not str(consigne).strip():
            raise ToolError("un problème bloquant ou majeur doit avoir une consigne de correction précise")
        self.findings.append({"gravite": gravite, "cible_site": cible_site, "cible_champ": cible_champ,
                              "constat": constat.strip(), "preuve": preuve.strip(), "consigne": str(consigne).strip(),
                              "source": "agent"})
        missing = self.uncovered()
        return {"constats_enregistres": len(self.findings), "cibles_non_encore_verifiees": missing}

    def uncovered(self) -> list[str]:
        """Sites (et synthèse) pour lesquels l'agent n'a encore rien consigné."""
        covered = {f["cible_site"] for f in self.findings if f.get("source") == "agent"}
        return [t for t in self.sites + ["synthese"] if t not in covered]

    def rule_verdict(self) -> dict:
        """Verdict calculé par les règles, quand l'agent n'a pas conclu."""
        failed = [c for c in self.checks if not c["ok"]]
        levels = {f["gravite"] for f in self.findings}
        verdict = ("a_refaire" if failed or "bloquant" in levels else
                   "valide_avec_reserves" if "majeur" in levels else "valide")
        return {"verdict": verdict, "synthese": "VERDICT AUTOMATIQUE (l'agent n'a pas conclu) : calculé à partir des "
                f"contrôles ({len(failed)} en échec) et des constats ({sorted(levels) or 'aucun'})."}

    def conclude(self, verdict: str, synthese: str) -> dict:
        if verdict not in VERDICTS:
            raise ToolError(f"verdict doit valoir : {VERDICTS}")
        if self.report_md is not None and not self.report_read:
            raise ToolError("lire le rapport (read_report) avant de conclure")
        if self.report_md is not None and self.uncovered():
            raise ToolError(f"vérification incomplète : consigne au moins un point (conforme ou problème) pour "
                            f"{self.uncovered()} avant de conclure")
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
        "name": "site_facts", "description": "Chiffres recalculés d'un site à partir des données (parts des thèmes, "
                                            "paires cause -> conséquence, critères du segment), à côté de ce que le "
                                            "rapport affirme pour ce site et de ses mesures.",
        "parameters": {"type": "object", "properties": {"site": {"type": "string"}},
                       "required": ["site"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "theme_facts", "description": "Contenu réel d'un thème de causes ou de conséquences : nombre de "
                                             "phrases et de rôles, sites, exemples de phrases.",
        "parameters": {"type": "object", "properties": {"theme_id": {"type": "string"}},
                       "required": ["theme_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "record_finding", "description": "Consigne un point vérifié : 'conforme' s'il est correct, sinon "
                                                "un problème (bloquant, majeur, mineur), avec sa preuve.",
        "parameters": {"type": "object", "properties": {
            "gravite": {"type": "string", "enum": GRAVITES},
            "cible_site": {"type": "string", "description": "site concerné, ou « synthese », ou « general »"},
            "cible_champ": {"type": "string", "enum": FIELDS},
            "constat": {"type": "string"}, "preuve": {"type": "string", "description": "chiffres ou extrait"},
            "consigne": {"type": "string", "description": "correction précise à faire (obligatoire si bloquant ou majeur)"}},
            "required": ["gravite", "cible_site", "cible_champ", "constat", "preuve", "consigne"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "conclude", "description": "Outil final : verdict et synthèse de la vérification.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": VERDICTS}, "synthese": {"type": "string"}},
            "required": ["verdict", "synthese"], "additionalProperties": False}}},
]
