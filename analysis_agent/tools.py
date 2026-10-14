"""Les outils de l'agent d'analyse.

Chaque outil est une méthode de AnalysisWorkspace :
  - ses arguments sont bornés et vérifiés (valeur hors bornes -> refus motivé, rien n'est calculé) ;
  - il vérifie que les étapes préalables ont été faites (pas de clustering sans causes extraites...) ;
  - tous les chiffres qu'il renvoie sont calculés par le code, jamais par le LLM.

Deux clusterings par type d'installation, sur les phrases extraites par le LLM : les CAUSES
(pourquoi) et les CONSÉQUENCES (ce que l'incident a provoqué). Les causes inconnues (« cause non
précisée ») sont exclues du clustering des causes et comptées à part.

Outils, dans l'ordre d'utilisation :
    overview                 vue d'ensemble des données et profils de site
    extract_causes           cause et conséquence de chaque incident, par le LLM (cache partagé)
    run_clustering           un essai de clustering (UMAP + HDBSCAN) sur les causes ou les conséquences
    select_clustering        choix d'un essai, refusé s'il ne respecte pas les seuils de qualité
    topic_details            exemples et statistiques de clusters
    define_themes            range des clusters dans des thèmes nommés
    remove_theme             défait un thème
    site_evidence            points distinctifs et statistiques de chaque site
    assign_profiles          associe chaque site à un profil A, B, C (outil final de la phase profils)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from llm.agent import ToolError
from paths import CACHE_DIR
from topic_modeling.config import FRENCH_STOPWORDS, NGRAM_RANGE, UMAP_PARAMS

from . import facts as F
from .config import (CONSEQUENCE_WORDS, MERGE_SIMILARITY, MIN_CHOSEN_DBCV, ASSET_TYPES, DIMENSIONS, EXAMPLE_MAX_CHARS, EXTRA_STOPWORDS, MAX_CLUSTERING_RUNS,
                     MAX_THEME_NAME_CHARS, MAX_THEMES_PER_CALL, MAX_TOPICS_PER_CALL, MAX_UNCLASSIFIED_PCT,
                     MERGE_MIN_COHESION_RATIO, MIN_CLUSTER_SIZE_RANGE, MIN_STABILITY, N_DESCRIPTIONS,
                     N_EXAMPLES, N_TOPICS_RANGE, NEUTRAL_FACTORS, SITE_PROFILES, STABILITY_SEEDS)

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

    vectorizer = CountVectorizer(stop_words=FRENCH_STOPWORDS + EXTRA_STOPWORDS, ngram_range=NGRAM_RANGE,
                                 token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    counts = vectorizer.fit_transform(texts)
    clusters = sorted(set(labels))
    tf = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in clusters]).astype(float)
    idf = np.log(1 + tf.sum(axis=1).mean() / np.maximum(tf.sum(axis=0), 1))
    scores = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1) * idf
    vocab = vectorizer.get_feature_names_out()
    return {c: [vocab[i] for i in scores[k].argsort()[::-1][:n]] for k, c in enumerate(clusters)}


GENERIC = re.compile(r"\b(?:" + "|".join(map(re.escape, EXTRA_STOPWORDS)) + r")\b", re.IGNORECASE)


def normalize_phrase(text: str) -> str:
    """Phrase utilisée pour le clustering : sans les nombres et leurs unités (« arrêt de 3.2 heures »)
    ni les mots génériques (« éolienne », « pendant »), qui sinon forment de faux clusters."""
    text = re.sub(r"\d+(?:[.,]\d+)?\s*(?:heures?|h\b|€|euros?|%|jours?|minutes?|min\b)?", " ", str(text))
    text = GENERIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,.;:-") or "?"


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
        self.profiles: dict[str, dict] | None = None  # site -> {"profil", "raisonnement"} (assign_profiles)
        self.report: dict | None = None         # contenu du rapport (rédigé ensuite par writing.py)
        self._theme_count = 0

    @property
    def tools(self) -> dict:
        return {"overview": self.overview, "extract_causes": self.extract_causes,
                "run_clustering": self.run_clustering, "select_clustering": self.select_clustering,
                "topic_details": self.topic_details, "define_themes": self.define_themes,
                "define_theme": self.define_theme, "compare_clusters": self.compare_clusters,
                "merge_themes": self.merge_themes,
                "remove_theme": self.remove_theme, "site_evidence": self.site_evidence,
                "assign_profiles": self.assign_profiles}

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
            if dimension == "cause":  # causes inconnues ou incertaines : exclues du clustering, comptées à part
                phrases = phrases[~self.uncertain_or_unknown(phrases.index)]
            emb = embed_texts([normalize_phrase(p) for p in phrases], CACHE_DIR)
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            self.vectors[k] = (phrases.index.to_numpy(), emb)
        return self.vectors[k]

    def uncertain_or_unknown(self, positions) -> np.ndarray:
        """Incidents sans cause exploitable : inconnue, ou inférée avec une confiance faible."""
        c = self.causes.loc[positions]
        return ((c["cause_type"] == "inconnue") | ((c["cause_type"] == "inférée") & (c["confiance"] == "faible"))).to_numpy()

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
            out[f"theme_{d}"] = out[f"theme_{d}_id"].map(names).fillna(F.UNCLASSIFIED)
        if "cause_type" in out:
            out.loc[(out["cause_type"] == "inférée") & (out["confiance"] == "faible"), "theme_cause"] = F.UNCERTAIN
            out.loc[out["cause_type"] == "inconnue", "theme_cause"] = F.UNKNOWN
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
                "type_de_cause_pct": percent(ok["cause_type"]),
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
            "qualite": None,
            "essais_restants": MAX_CLUSTERING_RUNS - done - 1,
            "clusters": [{"cluster": int(t), "incidents": int(n), "mots_cles": keywords[t][:5]} for t, n in sizes.items()],
        }
        summary["qualite"] = F.clustering_quality(summary)
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
        better = [i for i, r in self.runs.items() if r["key"] == k and r["summary"]["conforme_aux_seuils"]
                  and r["summary"]["dbcv"] >= MIN_CHOSEN_DBCV]
        if summary["dbcv"] < MIN_CHOSEN_DBCV and better:
            raise ToolError(f"{run_id} a une séparation faible (DBCV {summary['dbcv']} < {MIN_CHOSEN_DBCV}) alors que ces "
                            f"essais conformes font mieux : {better}. Choisis l'un d'eux (le plus détaillé si possible).")
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
            if dimension == "cause":
                detail["type_de_cause_pct"] = percent(sub["cause_type"])
            else:
                detail["categories_consequence_pct"] = percent(sub["categorie_consequence"], top=3)
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
                if dimension == "cause" and re.search(CONSEQUENCE_WORDS, name, re.IGNORECASE):
                    raise ToolError(f"nom refusé : « {name} » décrit une conséquence (arrêt, perte, dégât...). Un thème "
                                    "de cause ne nomme que la cause (ex. « Vent fort / tempête »).")
                description = str(theme.get("description") or "").strip()[:600]
                justification = str(theme.get("justification") or "").strip()[:1500]
                topics = list(dict.fromkeys(check_list(theme.get("topics"), 40, "topics")))
                unknown = [t for t in topics if t not in known]
                if unknown:
                    raise ToolError(f"clusters inconnus : {unknown} (existants : {known})")
                # Clusters déjà rangés : ignorés (pas d'erreur), pour qu'un agent qui redéclare ses thèmes
                # ne tourne pas en rond.
                already = [t for t in topics if self.theme_of(asset_type, dimension, t)]
                topics = [t for t in topics if t not in already]
                if not topics:
                    results.append({"name": name, "ok": True, "note": f"clusters {already} déjà rangés : rien à faire"})
                    continue
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
                if len(topics) > 1 and dimension == "consequence":  # conséquences : même catégorie normalisée
                    dominant = {t: self.causes.loc[positions[labels == t], "categorie_consequence"].mode().iat[0]
                                for t in topics}
                    if len(set(dominant.values())) > 1:
                        raise ToolError(f"fusion refusée : ces clusters ont des catégories de conséquence différentes "
                                        f"{dominant}. Ne réunis que des clusters de même catégorie.")
                if len(topics) > 1 and dimension == "cause":  # causes : les phrases réunies doivent rester semblables
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
                results.append({"name": name, "ok": True, "theme_id": theme_id, "incidents": int(mask.sum()),
                                **({"ignores_deja_ranges": already} if already else {})})
            except ToolError as e:
                results.append({"name": theme.get("name") if isinstance(theme, dict) else None, "ok": False,
                                "refus": str(e)})
        return {"resultats": results, "restants_sans_theme": self.unthemed(asset_type, dimension)}

    def define_theme(self, asset_type: str, dimension: str, name: str, topics: list[int], description: str = "") -> dict:
        """Un seul thème par appel (schéma simple, plus fiable avec un petit modèle)."""
        result = self.define_themes(asset_type, dimension, [{"name": name, "topics": topics, "description": description}])
        outcome = result["resultats"][0]
        if not outcome["ok"]:
            raise ToolError(outcome["refus"] + f" (clusters encore sans thème : {result['restants_sans_theme']})")
        return {**outcome, "restants_sans_theme": result["restants_sans_theme"]}

    def dominant(self, asset_type: str, dimension: str, mask: np.ndarray) -> str:
        """Facteur dominant (causes, hors « indéterminé ») ou catégorie dominante (conséquences)."""
        positions, _ = self.phrase_vectors(asset_type, dimension)
        column = "facteur_principal" if dimension == "cause" else "categorie_consequence"
        values = self.causes.loc[positions[mask], column].dropna()
        values = values[values != "indéterminé"]
        return values.mode().iat[0] if len(values) else ""

    def centroid(self, asset_type: str, dimension: str, mask: np.ndarray) -> np.ndarray:
        _, vectors = self.phrase_vectors(asset_type, dimension)
        center = vectors[mask].mean(axis=0)
        return center / max(np.linalg.norm(center), 1e-12)

    def similar_cluster_pairs(self, asset_type: str, dimension: str) -> list[tuple[int, int, float]]:
        """Paires de clusters à fusionner : centres très proches ET même facteur / catégorie dominant."""
        labels = self.chosen(asset_type, dimension)["labels"]
        topics = self.topic_ids(asset_type, dimension)
        info = {t: (self.centroid(asset_type, dimension, labels == t), self.dominant(asset_type, dimension, labels == t))
                for t in topics}
        pairs = [(a, b, round(float(info[a][0] @ info[b][0]), 3)) for i, a in enumerate(topics) for b in topics[i + 1:]
                 if info[a][1] and info[a][1] == info[b][1]]
        return [p for p in pairs if p[2] >= MERGE_SIMILARITY]

    def duplicate_themes(self, asset_type: str, dimension: str) -> list[tuple[str, str, float]]:
        """Thèmes qui décrivent la même chose (même règle que les clusters) : à fusionner."""
        labels = self.chosen(asset_type, dimension)["labels"]
        themes = {tid: th for tid, th in self.themes.items() if th["asset_type"] == asset_type and th["dimension"] == dimension}
        info = {tid: (self.centroid(asset_type, dimension, np.isin(labels, th["topics"])),
                      self.dominant(asset_type, dimension, np.isin(labels, th["topics"]))) for tid, th in themes.items()}
        ids = sorted(themes)
        return [(a, b, round(float(info[a][0] @ info[b][0]), 3)) for i, a in enumerate(ids) for b in ids[i + 1:]
                if info[a][1] and info[a][1] == info[b][1] and float(info[a][0] @ info[b][0]) >= MERGE_SIMILARITY]

    def compare_clusters(self, asset_type: str, dimension: str, a: int, b: int) -> dict:
        """Les clusters a et b décrivent-ils la même chose ? Similarité des centres + facteur dominant."""
        self.check_args(asset_type, dimension)
        known = self.topic_ids(asset_type, dimension)
        check_choice(a, known, "a")
        check_choice(b, known, "b")
        labels = self.chosen(asset_type, dimension)["labels"]
        column = DIMENSIONS[dimension]
        positions, _ = self.phrase_vectors(asset_type, dimension)
        sim = float(self.centroid(asset_type, dimension, labels == a) @ self.centroid(asset_type, dimension, labels == b))
        da, db = self.dominant(asset_type, dimension, labels == a), self.dominant(asset_type, dimension, labels == b)
        advice = ("fusion recommandée" if da == db and sim >= MERGE_SIMILARITY else
                  "fusion déconseillée : facteurs différents" if da != db else "fusion possible si les phrases le confirment")
        return {"similarite": round(sim, 3), "seuil": MERGE_SIMILARITY, "dominant": {a: da, b: db}, "avis": advice,
                "exemples": {t: self.causes.loc[positions[labels == t], column].head(3).tolist() for t in (a, b)}}

    def merge_themes(self, asset_type: str, dimension: str, theme_ids: list[str], name: str) -> dict:
        """Fusionne des thèmes déjà créés en un seul (pour supprimer un doublon)."""
        self.check_args(asset_type, dimension)
        theme_ids = check_list(theme_ids, 10, "theme_ids")
        own = [tid for tid, th in self.themes.items() if th["asset_type"] == asset_type and th["dimension"] == dimension]
        for tid in theme_ids:
            check_choice(tid, own, "theme_id")
        if len(set(theme_ids)) < 2:
            raise ToolError("il faut au moins 2 thèmes à fusionner")
        topics = [t for tid in theme_ids for t in self.themes[tid]["topics"]]
        backup = {tid: self.themes.pop(tid) for tid in theme_ids}
        result = self.define_themes(asset_type, dimension, [{"name": name, "topics": topics,
                                                             "description": backup[theme_ids[0]]["description"]}])
        outcome = result["resultats"][0]
        if not outcome["ok"]:
            self.themes.update(backup)  # fusion refusée : on remet les thèmes d'origine
            raise ToolError(outcome["refus"])
        return {**outcome, "doublons_restants": self.duplicate_themes(asset_type, dimension)}

    def remove_theme(self, theme_id: str) -> dict:
        check_choice(theme_id, list(self.themes), "theme_id")
        theme = self.themes.pop(theme_id)
        return {"clusters_liberes": theme["topics"],
                "restants_sans_theme": self.unthemed(theme["asset_type"], theme["dimension"])}

    def site_evidence(self, asset_type: str) -> dict:
        """Ce qui distingue chaque site des autres sites du même type (calculé en Python, facts.py)."""
        self.check_args(asset_type)
        for d in DIMENSIONS:
            self.chosen(asset_type, d)
        self.require_themes(asset_type)
        table = self.incidents_table()
        table = table[table["asset_type"] == asset_type]
        indicators = F.profile_indicators(table)
        sites = {}
        for site, g in table.groupby("site"):
            sites[site] = {
                "points_distinctifs": [F.point_text(p) for p in F.distinctive_points(indicators, site)]
                or ["aucun critère ne distingue nettement ce site"],
                "causes_connues_les_plus_frequentes": [f"{c['cause']} ({c['part_incidents_pct']} % des incidents)"
                                                      for c in F.dominant_causes(g)],
                "cause_non_precisee_pct": F.unknown_share(g),
            }
        return {"asset_type": asset_type, "profils_possibles": SITE_PROFILES[asset_type],
                "criteres_par_site": F.rank_text(indicators), "sites": sites,
                "compatibilite_profils": F.profile_compatibility(indicators, asset_type),
                "meilleures_attributions": F.best_assignments(indicators, asset_type)}

    def assign_profiles(self, assignments: list[dict], raisonnement: str) -> dict:
        """Outil final de la phase profils : un profil A, B, C par site, chacun une fois par type."""
        raisonnement = check_text(raisonnement, "raisonnement", 4000)
        expected = {s: a for a in ASSET_TYPES for s in self.sites[a]}
        if not isinstance(assignments, list) or not all(isinstance(x, dict) for x in assignments) \
                or sorted(x.get("site", "") for x in assignments) != sorted(expected):
            raise ToolError(f"il faut exactement une entrée par site : {sorted(expected)}")
        table = self.incidents_table()
        for asset_type in ASSET_TYPES:
            profiles = sorted(str(x.get("profil")) for x in assignments if expected[x["site"]] == asset_type)
            if profiles != sorted(SITE_PROFILES[asset_type]):
                raise ToolError(f"{asset_type} : chaque profil A, B, C doit être attribué à exactement un site "
                                f"(reçu : {profiles})")
            indicators = F.profile_indicators(table[table["asset_type"] == asset_type])
            chosen = {x["site"]: x["profil"] for x in assignments if expected[x["site"]] == asset_type}
            best = F.best_assignments(indicators, asset_type)
            score, top = F.assignment_score(indicators, asset_type, chosen), F.assignment_score(indicators, asset_type, best[0])
            if score < top:
                raise ToolError(f"{asset_type} : cette attribution respecte {score} attentes des profils, alors que "
                                f"{best} en respectent {top}. Choisis l'une de ces meilleures attributions.")
        self.profiles = {x["site"]: {"profil": x["profil"], "raisonnement": raisonnement} for x in assignments}
        return {"message": "profils enregistrés"}


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
              "required": ["name", "topics"],
              "properties": {"name": {"type": "string", "description": f"{MAX_THEME_NAME_CHARS} caractères maximum"},
                             "topics": {"type": "array", "items": {"type": "integer"}},
                             "description": {"type": "string", "description": "1 à 2 phrases"},
                             "justification": {"type": "string"}}}}}),
    tool("define_theme", "Crée UN thème nommé à partir d'un ou plusieurs clusters (plusieurs = fusion). Appelle-le une "
                         "fois par thème ; plusieurs appels par tour sont possibles.",
         {"asset_type": ASSET, "dimension": DIMENSION,
          "name": {"type": "string", "description": f"{MAX_THEME_NAME_CHARS} caractères maximum"},
          "topics": {"type": "array", "items": {"type": "integer"}},
          "description": {"type": "string", "description": "facultatif, 1 phrase"}}, required=["asset_type", "dimension", "name", "topics"]),
    tool("compare_clusters", "Les clusters a et b décrivent-ils la même chose ? Renvoie la similarité de leurs centres, "
                             "leur facteur (ou catégorie) dominant, des exemples et un avis de fusion.",
         {"asset_type": ASSET, "dimension": DIMENSION, "a": {"type": "integer"}, "b": {"type": "integer"}}),
    tool("merge_themes", "Fusionne plusieurs thèmes déjà créés en un seul (pour supprimer un doublon).",
         {"asset_type": ASSET, "dimension": DIMENSION, "theme_ids": {"type": "array", "items": {"type": "string"}},
          "name": {"type": "string"}}),
    tool("remove_theme", "Supprime un thème pour libérer ses clusters.", {"theme_id": {"type": "string"}}),
    tool("site_evidence", "Pour chaque site : points distinctifs (critères où il se distingue nettement des autres "
                          "sites), causes connues les plus fréquentes, part de causes non précisées ; et les critères "
                          "des profils mesurés sur chaque site.", {"asset_type": ASSET}),
    tool("assign_profiles", "Outil final : associe chaque site à un profil A, B ou C (chaque profil une fois par type).",
         {"assignments": {"type": "array", "items": {
             "type": "object", "additionalProperties": False, "required": ["site", "profil"],
             "properties": {"site": {"type": "string"}, "profil": {"type": "string", "enum": ["A", "B", "C"]}}}},
          "raisonnement": {"type": "string", "description": "pourquoi chaque site correspond à son profil"}}),
]
