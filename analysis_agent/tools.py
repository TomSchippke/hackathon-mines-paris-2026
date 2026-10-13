"""Les outils de l'agent d'analyse.

Chaque outil est une méthode de AnalysisWorkspace :
  - ses arguments sont bornés et vérifiés (valeur hors bornes -> refus motivé, rien n'est calculé) ;
  - il vérifie que les étapes préalables ont été faites (pas de clustering sans causes extraites...) ;
  - tous les chiffres qu'il renvoie sont calculés par le code, jamais par le LLM.

Deux clusterings par type d'installation, sur les phrases extraites par le LLM : les CAUSES
(pourquoi) et les CONSÉQUENCES (ce que l'incident a provoqué). liens_cause_consequence les relie.

Outils, dans l'ordre d'utilisation :
    overview                 vue d'ensemble des données et profils de site
    extract_causes           cause et conséquence de chaque incident, par le LLM (cache partagé)
    run_clustering           un essai de clustering (UMAP + HDBSCAN) sur les causes ou les conséquences
    select_clustering        choix d'un essai, refusé s'il ne respecte pas les seuils de qualité
    topic_details            exemples et statistiques de clusters
    define_themes            range des clusters dans des thèmes nommés
    remove_theme             défait un thème
    liens_cause_consequence  quelles causes mènent à quelles conséquences
    site_evidence            statistiques par site pour l'association aux profils et la remédiation
    write_report             rédaction du rapport final (outil final)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from llm.agent import ToolError
from paths import CACHE_DIR
from topic_modeling.config import FRENCH_STOPWORDS, NGRAM_RANGE, UMAP_PARAMS

from .config import (ACTIONS_PER_SITE, ASSET_TYPES, DIMENSIONS, EXAMPLE_MAX_CHARS, MAX_CLUSTERING_RUNS,
                     MAX_THEME_NAME_CHARS, MAX_THEMES_PER_CALL, MAX_TOPICS_PER_CALL, MAX_UNCLASSIFIED_PCT,
                     MERGE_MIN_COHESION_RATIO, MIN_CLUSTER_SIZE_RANGE, MIN_STABILITY, N_DESCRIPTIONS,
                     N_EXAMPLES, N_TOPICS_RANGE, NEUTRAL_FACTORS, PROFILE_CRITERIA, SITE_PROFILES,
                     STABILITY_SEEDS, TOP_PAIRS)

SEASONS = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps", 5: "printemps",
           6: "été", 7: "été", 8: "été", 9: "automne", 10: "automne", 11: "automne"}
OTHER = {"cause": "consequence", "consequence": "cause"}


# ------------------------------------------------------------------------------------------------
# Petites fonctions de calcul et de vérification
# ------------------------------------------------------------------------------------------------

def percent(series: pd.Series, top: int | None = None) -> dict[str, float]:
    """Répartition en % des valeurs d'une colonne (les plus fréquentes d'abord)."""
    shares = (series.dropna().value_counts(normalize=True) * 100).round(1)
    return (shares.head(top) if top else shares).to_dict()


def keywords_by_cluster(texts: list[str], labels: np.ndarray, n: int = 8) -> dict[int, list[str]]:
    """Mots-clés de chaque cluster (c-TF-IDF, comme BERTopic) : fréquents dans le cluster, rares ailleurs."""
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words=FRENCH_STOPWORDS, ngram_range=NGRAM_RANGE,
                                 token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    counts = vectorizer.fit_transform(texts)
    clusters = sorted(set(labels))
    tf = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in clusters]).astype(float)
    idf = np.log(1 + tf.sum(axis=1).mean() / np.maximum(tf.sum(axis=0), 1))
    scores = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1) * idf
    vocab = vectorizer.get_feature_names_out()
    return {c: [vocab[i] for i in scores[k].argsort()[::-1][:n]] for k, c in enumerate(clusters)}


def normalize_phrase(text: str) -> str:
    """Phrase utilisée pour le clustering : sans les nombres ni leurs unités (« arrêt de 3.2 heures »
    -> « arrêt de »), qui sinon séparent des conséquences identiques."""
    text = re.sub(r"\d+(?:[.,]\d+)?\s*(?:heures?|h\b|€|euros?|%|jours?|minutes?|min\b)?", " ", str(text))
    return re.sub(r"\s+", " ", text).strip(" ,.;:-")


def check_choice(value, allowed, name: str):
    if value not in allowed:
        raise ToolError(f"{name} doit valoir l'une de ces valeurs : {list(allowed)} (reçu : {value!r})")
    return value


def check_int(value, low: int, high: int, name: str) -> int:
    if not isinstance(value, int) or not low <= value <= high:
        raise ToolError(f"{name} doit être un entier entre {low} et {high} (reçu : {value!r})")
    return value


def check_text(value, name: str, max_chars: int = 3000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} ne doit pas être vide")
    if len(value) > max_chars:
        raise ToolError(f"{name} : {max_chars} caractères maximum (reçu : {len(value)})")
    return value.strip()


def quality_problems(stability: float, unclassified: float, n_topics: int) -> list[str]:
    """Seuils de qualité non respectés par un essai de clustering (liste vide = conforme)."""
    problems = []
    if stability < MIN_STABILITY:
        problems.append(f"stabilité {stability:.3f} < {MIN_STABILITY} exigé")
    if unclassified > MAX_UNCLASSIFIED_PCT:
        problems.append(f"{unclassified} % de non classés > {MAX_UNCLASSIFIED_PCT} % autorisés : baisser min_cluster_size")
    if n_topics < N_TOPICS_RANGE[0]:
        problems.append(f"{n_topics} clusters < {N_TOPICS_RANGE[0]} exigés : baisser min_cluster_size")
    if n_topics > N_TOPICS_RANGE[1]:
        problems.append(f"{n_topics} clusters > {N_TOPICS_RANGE[1]} autorisés : augmenter min_cluster_size")
    return problems


def profile_indicators(incidents: pd.DataFrame) -> dict[str, dict]:
    """Critères des profils de site mesurés sur chaque site, avec le rang du site (1 = valeur la plus
    élevée parmi les sites du même type). Utilisé par l'agent d'analyse, le rapport et le vérificateur.

    `incidents` : incidents d'un type d'installation, avec les colonnes extraites par le LLM.
    Renvoie {critère: {site: {"valeur": ..., "rang": ...}}}.
    """
    factors = (incidents["facteur_principal"].fillna("") + " | " + incidents["facteurs_secondaires"].fillna(""))
    values: dict[str, pd.Series] = {}
    for criterion, names in PROFILE_CRITERIA.items():
        hit = factors.apply(lambda f: any(n in f for n in names))
        values[f"{criterion} (% incidents)"] = 100 * hit.groupby(incidents["site"]).mean()
    flag = lambda col: incidents[col].astype(str).str.lower().isin(["true", "1"])
    values["usure / vieillissement (% incidents)"] = 100 * flag("lie_vieillissement").groupby(incidents["site"]).mean()
    values["lié à la maintenance (% incidents)"] = 100 * flag("lie_maintenance").groupby(incidents["site"]).mean()
    values["incidents sévères (%)"] = 100 * (incidents["severity"] == "sévère").groupby(incidents["site"]).mean()
    values["coût moyen (€)"] = incidents.groupby("site")["estimated_cost_eur"].mean()
    out = {}
    for criterion, v in values.items():
        ranks = v.round(1).rank(ascending=False, method="min").astype(int)
        tied = v.round(1).nunique() == 1  # tous les sites égaux : aucun rang n'a de sens
        out[criterion] = {site: {"valeur": round(float(v[site]), 1), "rang": None if tied else int(ranks[site])}
                          for site in v.index}
    return out


def profile_indicators_text(indicators: dict[str, dict]) -> dict[str, str]:
    """Version compacte, facile à comparer : « Est-1 15.0 (1er) | Est-2 8.0 (2e) | Ouest-1 0.0 (3e) »."""
    def rank(r):
        return "égaux" if r is None else f"{r}{'er' if r == 1 else 'e'}"
    return {criterion: " | ".join(f"{site} {x['valeur']} ({rank(x['rang'])})"
                                  for site, x in sorted(sites.items(), key=lambda s: s[1]["rang"] or 0))
            for criterion, sites in indicators.items()}


def check_list(value, max_items: int, name: str) -> list:
    if not isinstance(value, list) or not 1 <= len(value) <= max_items:
        raise ToolError(f"{name} doit être une liste de 1 à {max_items} éléments")
    return value


# ------------------------------------------------------------------------------------------------
# Espace de travail de l'agent
# ------------------------------------------------------------------------------------------------

class AnalysisWorkspace:
    def __init__(self, incidents: pd.DataFrame):
        self.df = incidents.reset_index(drop=True).copy()
        self.df["saison"] = pd.to_datetime(self.df["date"], errors="coerce").dt.month.map(SEASONS)
        self.sites = {a: sorted(self.df.loc[self.df["asset_type"] == a, "site"].unique()) for a in ASSET_TYPES}
        self.causes = pd.DataFrame()            # analyses du LLM, index = position dans self.df
        self.vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # "type/dim" -> (positions, embeddings)
        self.reduced: dict[str, dict[int, np.ndarray]] = {}  # "type/dim" -> UMAP par graine (calculé une fois)
        self.runs: dict[str, dict] = {}         # essais de clustering, par run_id
        self.selected: dict[str, dict] = {}     # "type/dim" -> {"run_id", "justification"}
        self.themes: dict[str, dict] = {}       # theme_id -> thème
        self.report: dict | None = None         # contenu rédigé par l'agent (write_report)
        self._theme_count = 0

    @property
    def tools(self) -> dict:
        return {"overview": self.overview, "extract_causes": self.extract_causes,
                "run_clustering": self.run_clustering, "select_clustering": self.select_clustering,
                "topic_details": self.topic_details, "define_themes": self.define_themes,
                "remove_theme": self.remove_theme, "liens_cause_consequence": self.liens_cause_consequence,
                "site_evidence": self.site_evidence, "write_report": self.write_report}

    # ---- état et vérifications -----------------------------------------------------------------

    @staticmethod
    def key(asset_type: str, dimension: str) -> str:
        return f"{asset_type}/{dimension}"

    def check_args(self, asset_type: str, dimension: str | None = None) -> None:
        check_choice(asset_type, ASSET_TYPES, "asset_type")
        if dimension is not None:
            check_choice(dimension, list(DIMENSIONS), "dimension")

    def positions(self, asset_type: str) -> np.ndarray:
        return np.flatnonzero(self.df["asset_type"].to_numpy() == asset_type)

    def require_causes(self, asset_type: str) -> None:
        if not set(self.positions(asset_type)) <= set(self.causes.index):
            raise ToolError(f"causes non extraites pour {asset_type} : appeler extract_causes d'abord")

    def phrase_vectors(self, asset_type: str, dimension: str) -> tuple[np.ndarray, np.ndarray]:
        """Positions des incidents ayant une phrase extraite pour cette dimension, et leurs embeddings."""
        from topic_modeling.embeddings import embed_texts

        k = self.key(asset_type, dimension)
        if k not in self.vectors:
            self.require_causes(asset_type)
            column = DIMENSIONS[dimension]
            phrases = self.causes.loc[self.positions(asset_type), column]
            phrases = phrases[phrases.notna() & (phrases.astype(str).str.strip() != "")]
            emb = embed_texts([normalize_phrase(p) for p in phrases], CACHE_DIR)
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            self.vectors[k] = (phrases.index.to_numpy(), emb)
        return self.vectors[k]

    def chosen(self, asset_type: str, dimension: str) -> dict:
        k = self.key(asset_type, dimension)
        if k not in self.selected:
            raise ToolError(f"aucun clustering choisi pour {k} : appeler select_clustering d'abord")
        return self.runs[self.selected[k]["run_id"]]

    def topic_ids(self, asset_type: str, dimension: str) -> list[int]:
        return sorted(int(t) for t in set(self.chosen(asset_type, dimension)["labels"]) if t != -1)

    def theme_of(self, asset_type: str, dimension: str, topic: int) -> str | None:
        return next((tid for tid, th in self.themes.items() if th["asset_type"] == asset_type
                     and th["dimension"] == dimension and topic in th["topics"]), None)

    def unthemed(self, asset_type: str, dimension: str) -> list[int]:
        return [t for t in self.topic_ids(asset_type, dimension) if self.theme_of(asset_type, dimension, t) is None]

    def require_themes(self, asset_type: str, dimensions=tuple(DIMENSIONS)) -> None:
        for d in dimensions:
            missing = self.unthemed(asset_type, d)
            if missing:
                raise ToolError(f"clusters sans thème pour {self.key(asset_type, d)} : {missing} (define_themes d'abord)")

    def cohesion(self, vectors: np.ndarray) -> float:
        """Similarité cosinus moyenne au centre du groupe (1 = phrases identiques)."""
        center = vectors.mean(axis=0)
        return float((vectors @ (center / max(np.linalg.norm(center), 1e-12))).mean())

    def incidents_table(self) -> pd.DataFrame:
        """Chaque incident avec ses clusters, ses thèmes (cause et conséquence) et son analyse."""
        out = self.df.join(self.causes)
        names = {tid: th["name"] for tid, th in self.themes.items()}
        for d in DIMENSIONS:
            out[f"cluster_{d}"] = pd.Series(dtype="Int64")
            out[f"theme_{d}_id"] = None
            for asset_type in ASSET_TYPES:
                k = self.key(asset_type, d)
                if k not in self.selected:
                    continue
                run = self.runs[self.selected[k]["run_id"]]
                out.loc[run["positions"], f"cluster_{d}"] = run["labels"]
                out.loc[run["positions"], f"theme_{d}_id"] = [
                    self.theme_of(asset_type, d, int(t)) if t != -1 else None for t in run["labels"]]
            out[f"theme_{d}"] = out[f"theme_{d}_id"].map(names)
        return out

    # ---- outils ----------------------------------------------------------------------------------

    def overview(self) -> dict:
        sites = (self.df.groupby(["asset_type", "site"])
                 .agg(incidents=("incident_id", "count"), cout_moyen_eur=("estimated_cost_eur", "mean"),
                      arret_moyen_h=("downtime_hours", "mean"))
                 .round(1).reset_index().to_dict(orient="records"))
        return {"incidents": len(self.df), "periode": [self.df["date"].min(), self.df["date"].max()],
                "sites": sites, "severite_pct": percent(self.df["severity"]),
                "profils_de_site": SITE_PROFILES,
                "bornes": {"min_cluster_size": MIN_CLUSTER_SIZE_RANGE,
                           "essais_par_type_et_dimension": MAX_CLUSTERING_RUNS},
                "seuils_qualite": {"stabilite_min": MIN_STABILITY, "non_classes_max_pct": MAX_UNCLASSIFIED_PCT,
                                   "nombre_clusters": N_TOPICS_RANGE}}

    def extract_causes(self, asset_type: str) -> dict:
        from cause_extraction.config import CAUSE_INCONNUE
        from cause_extraction.pipeline import analyse_incidents

        self.check_args(asset_type)
        todo = [p for p in self.positions(asset_type) if p not in self.causes.index]
        if todo:
            self.causes = pd.concat([self.causes, analyse_incidents(self.df.loc[todo])])
        sub = self.df.loc[self.positions(asset_type)].join(self.causes)
        ok = sub[sub["erreur_analyse"] == ""]
        examples = ok.sample(min(3, len(ok)), random_state=0)
        return {"incidents_analyses": len(ok), "echecs": int(len(sub) - len(ok)),
                "cause_non_precisee_pct": round(100 * float((ok["cause_racine"] == CAUSE_INCONNUE).mean()), 1),
                "facteurs_par_site_pct": {site: percent(g["facteur_principal"], top=5) for site, g in ok.groupby("site")},
                "exemples": examples[["cause_racine", "consequence"]].to_dict(orient="records")}

    def run_clustering(self, asset_type: str, dimension: str, min_cluster_size: int) -> dict:
        from hdbscan import HDBSCAN
        from sklearn.metrics import adjusted_rand_score
        from umap import UMAP

        self.check_args(asset_type, dimension)
        check_int(min_cluster_size, *MIN_CLUSTER_SIZE_RANGE, "min_cluster_size")
        k = self.key(asset_type, dimension)
        run_id = f"{asset_type}_{dimension}_mcs{min_cluster_size}"
        if run_id in self.runs:
            return {**self.runs[run_id]["summary"], "note": "essai déjà fait, résultat rappelé (pas de budget consommé)"}
        done = sum(r["key"] == k for r in self.runs.values())
        if done >= MAX_CLUSTERING_RUNS:
            raise ToolError(f"budget épuisé : {MAX_CLUSTERING_RUNS} essais déjà faits pour {k}. "
                            f"Choisir parmi : {[i for i, r in self.runs.items() if r['key'] == k]}")

        positions, vectors = self.phrase_vectors(asset_type, dimension)
        if k not in self.reduced:  # la réduction UMAP ne dépend pas de min_cluster_size
            self.reduced[k] = {s: UMAP(**{**UMAP_PARAMS, "random_state": s}).fit_transform(vectors)
                               for s in STABILITY_SEEDS}
        fits = {s: HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_method="eom",
                           gen_min_span_tree=True).fit(x) for s, x in self.reduced[k].items()}
        main = fits[STABILITY_SEEDS[0]]
        labels = main.labels_
        stability = float(np.mean([adjusted_rand_score(labels, fits[s].labels_) for s in STABILITY_SEEDS[1:]]))
        sizes = pd.Series(labels[labels != -1]).value_counts()
        phrases = [normalize_phrase(p) for p in self.causes.loc[positions, DIMENSIONS[dimension]]]
        keywords = keywords_by_cluster(phrases, labels)

        n_topics = len(sizes)
        unclassified = round(100 * float((labels == -1).mean()), 1)
        summary = {
            "run_id": run_id, "asset_type": asset_type, "dimension": dimension,
            "min_cluster_size": min_cluster_size, "nombre_clusters": n_topics, "non_classes_pct": unclassified,
            "plus_gros_cluster_pct": round(100 * sizes.max() / len(labels), 1) if n_topics else None,
            "dbcv": round(float(main.relative_validity_), 3), "stabilite": round(stability, 3),
            "conforme_aux_seuils": not quality_problems(stability, unclassified, n_topics),
            "seuils_non_respectes": quality_problems(stability, unclassified, n_topics),
            "essais_restants": MAX_CLUSTERING_RUNS - done - 1,
            "clusters": [{"cluster": int(t), "incidents": int(n), "mots_cles": keywords[t][:5]} for t, n in sizes.items()],
        }
        self.runs[run_id] = {"key": k, "asset_type": asset_type, "dimension": dimension, "positions": positions,
                             "labels": labels, "keywords": keywords, "summary": summary}
        return summary

    def select_clustering(self, asset_type: str, dimension: str, run_id: str, justification: str) -> dict:
        self.check_args(asset_type, dimension)
        justification = check_text(justification, "justification")
        k = self.key(asset_type, dimension)
        if run_id not in self.runs or self.runs[run_id]["key"] != k:
            raise ToolError(f"essai inconnu pour {k} : {run_id}. Essais : {[i for i, r in self.runs.items() if r['key'] == k]}")
        summary = self.runs[run_id]["summary"]
        if not summary["conforme_aux_seuils"]:
            conform = [i for i, r in self.runs.items() if r["key"] == k and r["summary"]["conforme_aux_seuils"]]
            left = MAX_CLUSTERING_RUNS - sum(r["key"] == k for r in self.runs.values())
            raise ToolError(f"{run_id} n'est pas conforme : {' ; '.join(summary['seuils_non_respectes'])}. "
                            + (f"Essais conformes disponibles : {conform}." if conform else
                               f"Aucun essai conforme : relancer run_clustering avec une autre valeur de "
                               f"min_cluster_size ({left} essais restants)."))
        removed = [tid for tid, th in self.themes.items() if th["asset_type"] == asset_type and th["dimension"] == dimension]
        for tid in removed:  # des thèmes faits sur un autre clustering n'ont plus de sens
            del self.themes[tid]
        self.selected[k] = {"run_id": run_id, "justification": justification}
        return {"run_id": run_id, "clusters_a_ranger": self.topic_ids(asset_type, dimension), "themes_supprimes": removed}

    def topic_details(self, asset_type: str, dimension: str, topics: list[int]) -> dict:
        self.check_args(asset_type, dimension)
        run = self.chosen(asset_type, dimension)
        known = self.topic_ids(asset_type, dimension)
        topics = check_list(topics, MAX_TOPICS_PER_CALL, "topics")
        for t in topics:
            check_choice(t, known, "topic")
        positions, vectors = self.phrase_vectors(asset_type, dimension)
        column, other = DIMENSIONS[dimension], OTHER[dimension]
        details = []
        for t in topics:
            mask = run["labels"] == t
            pos = positions[mask]
            sub = self.df.loc[pos].join(self.causes)
            center = vectors[mask].mean(axis=0)
            typical = sub.iloc[np.argsort(-(vectors[mask] @ center))]  # les plus proches du centre d'abord
            detail = {
                "cluster": t, "incidents": len(sub), "mots_cles": run["keywords"][t],
                "cohesion": round(self.cohesion(vectors[mask]), 3), "theme": self.theme_of(asset_type, dimension, t),
                f"{column}_exemples": typical[column].head(N_EXAMPLES).tolist(),
                f"{DIMENSIONS[other]}_exemples": typical[DIMENSIONS[other]].head(2).tolist(),
                "description_exemple": [x[:EXAMPLE_MAX_CHARS] for x in typical["text"].head(N_DESCRIPTIONS)],
                "sites_pct": percent(sub["site"]), "severite_pct": percent(sub["severity"]),
                "saisons_pct": percent(sub["saison"], top=2), "meteo_pct": percent(sub["weather_condition"], top=3),
                "facteurs_pct": percent(sub["facteur_principal"], top=4),
                "cout_moyen_eur": round(float(sub["estimated_cost_eur"].mean())),
            }
            details.append(detail)
        return {"clusters": details, "restants_sans_theme": self.unthemed(asset_type, dimension)}

    def define_themes(self, asset_type: str, dimension: str, themes: list[dict]) -> dict:
        """Crée plusieurs thèmes ; chacun est accepté ou refusé indépendamment."""
        self.check_args(asset_type, dimension)
        themes = check_list(themes, MAX_THEMES_PER_CALL, "themes")
        known = self.topic_ids(asset_type, dimension)
        positions, vectors = self.phrase_vectors(asset_type, dimension)
        labels = self.chosen(asset_type, dimension)["labels"]
        results = []
        for theme in themes:
            try:
                if not isinstance(theme, dict):
                    raise ToolError("chaque thème doit être un objet {name, topics, description, justification}")
                name = check_text(theme.get("name"), "name", MAX_THEME_NAME_CHARS)
                description = check_text(theme.get("description"), "description", 600)
                justification = check_text(theme.get("justification"), "justification", 1500)
                topics = list(dict.fromkeys(check_list(theme.get("topics"), 40, "topics")))
                unknown = [t for t in topics if t not in known]
                if unknown:
                    raise ToolError(f"clusters inconnus : {unknown} (existants : {known})")
                taken = {t: self.theme_of(asset_type, dimension, t) for t in topics
                         if self.theme_of(asset_type, dimension, t)}
                if taken:
                    raise ToolError(f"clusters déjà dans un thème : {taken} (remove_theme pour les libérer)")
                mask = np.isin(labels, topics)
                cohesion = self.cohesion(vectors[mask])
                if len(topics) > 1 and dimension == "cause":  # fusion de causes : même facteur principal
                    dominant = {}
                    for t in topics:
                        f = self.causes.loc[positions[labels == t], "facteur_principal"].value_counts()
                        f = f.drop(index=list(NEUTRAL_FACTORS), errors="ignore")
                        if len(f):
                            dominant[t] = f.index[0]
                    if len(set(dominant.values())) > 1:
                        raise ToolError(f"fusion refusée : ces clusters ont des facteurs principaux différents "
                                        f"{dominant}. Fais un thème par facteur.")
                if len(topics) > 1:  # fusion : les phrases réunies doivent rester semblables
                    reference = np.average([self.cohesion(vectors[labels == t]) for t in topics],
                                           weights=[(labels == t).sum() for t in topics])
                    if cohesion < MERGE_MIN_COHESION_RATIO * reference:
                        raise ToolError(f"fusion refusée : cohésion {cohesion:.3f} contre {reference:.3f} dans les "
                                        f"clusters d'origine (minimum {MERGE_MIN_COHESION_RATIO:.0%}) : trop différents")
                self._theme_count += 1
                theme_id = f"{asset_type}_{dimension}_T{self._theme_count:02d}"
                self.themes[theme_id] = {"asset_type": asset_type, "dimension": dimension, "name": name,
                                         "description": description, "topics": topics, "justification": justification,
                                         "incidents": int(mask.sum()), "cohesion": round(cohesion, 3)}
                results.append({"name": name, "ok": True, "theme_id": theme_id, "incidents": int(mask.sum())})
            except ToolError as e:
                results.append({"name": theme.get("name") if isinstance(theme, dict) else None, "ok": False,
                                "refus": str(e)})
        return {"resultats": results, "restants_sans_theme": self.unthemed(asset_type, dimension)}

    def remove_theme(self, theme_id: str) -> dict:
        check_choice(theme_id, list(self.themes), "theme_id")
        theme = self.themes.pop(theme_id)
        return {"clusters_liberes": theme["topics"],
                "restants_sans_theme": self.unthemed(theme["asset_type"], theme["dimension"])}

    def links_table(self, asset_type: str) -> pd.DataFrame:
        table = self.incidents_table()
        table = table[table["asset_type"] == asset_type].copy()
        for d in DIMENSIONS:
            table[f"theme_{d}"] = table[f"theme_{d}"].fillna("non classé")
        return table

    def liens_cause_consequence(self, asset_type: str) -> dict:
        self.check_args(asset_type)
        self.require_themes(asset_type)
        t = self.links_table(asset_type)
        pairs = (t.groupby(["theme_cause", "theme_consequence"])
                 .agg(incidents=("incident_id", "count"), cout_moyen_eur=("estimated_cost_eur", "mean"),
                      cout_total_eur=("estimated_cost_eur", "sum"))
                 .sort_values("incidents", ascending=False).head(TOP_PAIRS).reset_index())
        pairs["part_pct"] = (100 * pairs["incidents"] / len(t)).round(1)
        pairs["sites_pct"] = [percent(t[(t["theme_cause"] == r.theme_cause) & (t["theme_consequence"] == r.theme_consequence)]
                                      ["site"], top=3) for r in pairs.itertuples()]
        pairs[["cout_moyen_eur", "cout_total_eur"]] = pairs[["cout_moyen_eur", "cout_total_eur"]].round(0)
        return {
            "paires_principales": pairs.to_dict(orient="records"),
            "consequences_de_chaque_cause_pct": {c: percent(g["theme_consequence"], top=3)
                                                 for c, g in t.groupby("theme_cause")},
            "causes_de_chaque_consequence_pct": {c: percent(g["theme_cause"], top=3)
                                                 for c, g in t.groupby("theme_consequence")},
        }

    def site_evidence(self, asset_type: str) -> dict:
        self.check_args(asset_type)
        self.require_themes(asset_type)
        t = self.links_table(asset_type)
        t = t[t["erreur_analyse"] == ""]
        sites = {}
        for site, g in t.groupby("site"):
            pairs = (g.groupby(["theme_cause", "theme_consequence"]).size().sort_values(ascending=False).head(3))
            sites[site] = {
                "incidents": len(g),
                "causes_pct": percent(g["theme_cause"], top=6),
                "consequences_pct": percent(g["theme_consequence"], top=4),
                "facteurs_pct": percent(g["facteur_principal"], top=5),
                "paires_principales": [f"{c} -> {q} ({n})" for (c, q), n in pairs.items()],
                "lie_vieillissement_pct": round(100 * float(g["lie_vieillissement"].astype(bool).mean()), 1),
                "lie_maintenance_pct": round(100 * float(g["lie_maintenance"].astype(bool).mean()), 1),
                "severe_pct": round(100 * float((g["severity"] == "sévère").mean()), 1),
                "cout_moyen_eur": round(float(g["estimated_cost_eur"].mean())),
                "arret_moyen_h": round(float(g["downtime_hours"].mean()), 1),
                "meteo_pct": percent(g["weather_condition"], top=3),
                "actions_correctives_frequentes": g["action_corrective"].value_counts().head(4).index.tolist(),
            }
        return {"asset_type": asset_type, "profils_possibles": SITE_PROFILES[asset_type],
                "indicateurs_profils": profile_indicators_text(profile_indicators(t)), "sites": sites}

    def write_report(self, synthese: str, causes_racines: str, chaines_cause_consequence: str,
                     sites: list[dict], limites: str) -> dict:
        """Outil final. Le contenu est vérifié ici ; la mise en forme et les tableaux chiffrés sont
        ajoutés par le code (report.py)."""
        for asset_type in ASSET_TYPES:
            self.require_causes(asset_type)
            for d in DIMENSIONS:
                self.chosen(asset_type, d)
            self.require_themes(asset_type)
        text = {"synthese": check_text(synthese, "synthese", 4000),
                "causes_racines": check_text(causes_racines, "causes_racines", 6000),
                "chaines_cause_consequence": check_text(chaines_cause_consequence, "chaines_cause_consequence", 6000),
                "limites": check_text(limites, "limites", 3000)}

        expected = {s: a for a in ASSET_TYPES for s in self.sites[a]}
        if not isinstance(sites, list) or not all(isinstance(s, dict) for s in sites) \
                or sorted(s.get("site", "") for s in sites) != sorted(expected):
            raise ToolError(f"il faut exactement une entrée par site : {sorted(expected)}")
        for asset_type in ASSET_TYPES:
            profiles = sorted(str(s.get("profil")) for s in sites if expected[s["site"]] == asset_type)
            if profiles != sorted(SITE_PROFILES[asset_type]):
                raise ToolError(f"{asset_type} : chaque profil A, B, C doit être attribué à exactement un site "
                                f"(reçu : {profiles})")
        for s in sites:
            for field in ("justification_profil", "causes_dominantes"):
                s[field] = check_text(s.get(field), f"{s['site']}.{field}")
            actions = s.get("plan_remediation")
            if not isinstance(actions, list) or not ACTIONS_PER_SITE[0] <= len(actions) <= ACTIONS_PER_SITE[1]:
                raise ToolError(f"{s['site']} : plan_remediation doit contenir entre {ACTIONS_PER_SITE[0]} et "
                                f"{ACTIONS_PER_SITE[1]} actions")
            for a in actions:
                if not isinstance(a, dict):
                    raise ToolError(f"{s['site']} : chaque action doit être un objet {{action, priorite, justification}}")
                check_choice(a.get("priorite"), ["haute", "moyenne", "basse"], f"{s['site']}.priorite")
                a["action"] = check_text(a.get("action"), f"{s['site']}.action", 400)
                a["justification"] = check_text(a.get("justification"), f"{s['site']}.justification", 600)
        self.report = {**text, "sites": sorted(sites, key=lambda s: (expected[s["site"]], s["site"]))}
        return {"message": "rapport enregistré"}


ASSET = {"type": "string", "enum": ASSET_TYPES}
DIMENSION = {"type": "string", "enum": list(DIMENSIONS), "description": "cause (pourquoi) ou consequence (ce que "
                                                                        "l'incident a provoqué)"}


def tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or list(properties),
                       "additionalProperties": False}}}


TOOLS = [
    tool("overview", "Vue d'ensemble : incidents par site, sévérité, profils de site du sujet, bornes et seuils.", {}),
    tool("extract_causes", "Fait extraire par le LLM la cause (pourquoi) et la conséquence (ce qui a été provoqué) "
                           "de chaque incident du type, avec facteur principal, usure, maintenance, action corrective.",
         {"asset_type": ASSET}),
    tool("run_clustering", "Un essai de clustering (UMAP + HDBSCAN) des causes ou des conséquences extraites. Renvoie "
                           "les scores (stabilité, non classés, DBCV, nombre de clusters) et les clusters avec mots-clés.",
         {"asset_type": ASSET, "dimension": DIMENSION,
          "min_cluster_size": {"type": "integer", "minimum": MIN_CLUSTER_SIZE_RANGE[0],
                               "maximum": MIN_CLUSTER_SIZE_RANGE[1],
                               "description": "taille minimale d'un cluster ; petit = clusters nombreux et fins"}}),
    tool("select_clustering", "Choisit l'essai de clustering à utiliser pour ce type et cette dimension.",
         {"asset_type": ASSET, "dimension": DIMENSION, "run_id": {"type": "string"}, "justification": {"type": "string"}}),
    tool("topic_details", f"Détail de 1 à {MAX_TOPICS_PER_CALL} clusters du clustering choisi : phrases typiques, "
                          "sites, sévérité, saisons, météo, facteurs, coût, cohésion (1 = phrases identiques).",
         {"asset_type": ASSET, "dimension": DIMENSION,
          "topics": {"type": "array", "items": {"type": "integer"}, "maxItems": MAX_TOPICS_PER_CALL}}),
    tool("define_themes", f"Range des clusters dans 1 à {MAX_THEMES_PER_CALL} thèmes nommés (un thème peut fusionner "
                          "plusieurs clusters ; fusion refusée si les phrases réunies sont trop différentes).",
         {"asset_type": ASSET, "dimension": DIMENSION,
          "themes": {"type": "array", "maxItems": MAX_THEMES_PER_CALL, "items": {
              "type": "object", "additionalProperties": False,
              "required": ["name", "topics", "description", "justification"],
              "properties": {"name": {"type": "string", "description": f"{MAX_THEME_NAME_CHARS} caractères maximum"},
                             "topics": {"type": "array", "items": {"type": "integer"}},
                             "description": {"type": "string", "description": "1 à 2 phrases"},
                             "justification": {"type": "string"}}}}}),
    tool("remove_theme", "Supprime un thème pour libérer ses clusters.", {"theme_id": {"type": "string"}}),
    tool("liens_cause_consequence", "Croise les thèmes de causes et de conséquences : paires les plus fréquentes "
                                    "(coût, sites), conséquences de chaque cause, causes de chaque conséquence.",
         {"asset_type": ASSET}),
    tool("site_evidence", "Statistiques par site (thèmes de causes et de conséquences, facteurs, paires, usure, "
                          "maintenance, sévérité, coût, météo, actions fréquentes) et profils possibles.",
         {"asset_type": ASSET}),
    tool("write_report", "Outil final : rédige le rapport d'analyse. Les tableaux chiffrés sont ajoutés automatiquement.",
         {"synthese": {"type": "string", "description": "résumé pour un responsable : 5 à 10 phrases"},
          "causes_racines": {"type": "string", "description": "causes racines principales, par type d'installation"},
          "chaines_cause_consequence": {"type": "string", "description": "quelles causes mènent à quelles "
                                                                         "conséquences, lesquelles coûtent le plus"},
          "sites": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["site", "profil", "justification_profil", "causes_dominantes", "plan_remediation"],
              "properties": {
                  "site": {"type": "string"}, "profil": {"type": "string", "enum": ["A", "B", "C"]},
                  "justification_profil": {"type": "string", "description": "faits observés qui correspondent au profil"},
                  "causes_dominantes": {"type": "string"},
                  "plan_remediation": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["action", "priorite", "justification"],
                      "properties": {"action": {"type": "string"},
                                     "priorite": {"type": "string", "enum": ["haute", "moyenne", "basse"]},
                                     "justification": {"type": "string"}}}}}}},
          "limites": {"type": "string", "description": "limites de l'analyse et points à confirmer"}}),
]
