"""Les outils de l'agent d'analyse et l'espace de travail de la découverte.

Chaque outil est une méthode de AnalysisWorkspace :
  - ses arguments sont bornés et vérifiés (valeur hors bornes -> refus motivé, rien n'est calculé) ;
  - il vérifie que les étapes préalables ont été faites (pas de clustering sans causes extraites...) ;
  - tous les chiffres qu'il renvoie sont calculés par le code, jamais par le LLM.

Les phrases (self.phrases, §2.3 du plan) sont les causes racines, les causes contributives vérifiées et les
conséquences de chaque incident. Elles sont regroupées par clustering, puis chaque phrase reçoit un thème (ou
aucun) par affectation (pipeline, assignment.py). Les causes inconnues ou incertaines sont exclues et comptées à part.

Outils, dans l'ordre d'utilisation :
    overview                 vue d'ensemble des données et seuils
    extract_causes           cause, causes contributives et conséquence de chaque incident, par le LLM (cache partagé)
    run_clustering           un essai de clustering (UMAP + HDBSCAN) sur les causes ou les conséquences
    select_clustering        choix d'un essai, refusé s'il ne respecte pas les seuils de qualité
    topic_details            exemples et statistiques de clusters
    define_themes            range des clusters dans des thèmes nommés
    define_theme             un seul thème (schéma simple)
    compare_clusters         similarité de deux clusters
    merge_themes             fusionne des thèmes en doublon
    remove_theme             défait un thème
    heatmap_hotspots         cases les plus coûteuses de la heatmap causes x conséquences
    select_priority_chains   chaînes cause -> conséquence prioritaires (outil final de la phase priorités)
    nominal_evidence         par thème de conséquence : part à la sévérité la plus faible, coût, exemples
    flag_nominal             thèmes de conséquences qui sont des arrêts nominaux (outil final de la phase nominale)

Méthodes du pipeline (non exposées à l'agent) : build_phrases, theme_cards, apply_assignment, none_share,
none_phrases, recluster_none, set_active_run. Voir leurs docstrings pour l'ordre d'appel.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from framing.cadrage import current
from llm.agent import ToolError
from paths import CACHE_DIR
from topic_modeling.config import FRENCH_STOPWORDS, NGRAM_RANGE, UMAP_PARAMS

from . import facts as F
from .config import (CONSEQUENCE_WORDS, DIMENSIONS, EXAMPLE_MAX_CHARS, EXTRA_STOPWORDS, GENERIC_MIN_DF, HOTSPOTS_MAX,
                     MAX_CLUSTERING_RUNS, MAX_THEME_NAME_CHARS, MAX_THEMES_PER_CALL, MAX_TOPICS_PER_CALL,
                     MAX_UNCLASSIFIED_PCT, MERGE_MIN_COHESION_RATIO, MERGE_SIMILARITY, MIN_CHOSEN_DBCV,
                     MIN_CLUSTER_SIZE_RANGE, MIN_STABILITY, N_DESCRIPTIONS, N_EXAMPLES, N_TOPICS_RANGE,
                     NOMINAL_MIN_MILDEST_PCT, PRIORITY_CHAINS_RANGE, RECLUSTER_MIN_CLUSTER_SIZE, STABILITY_SEEDS)

SEASONS = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps", 5: "printemps",
           6: "été", 7: "été", 8: "été", 9: "automne", 10: "automne", 11: "automne"}
OTHER = {"cause": "consequence", "consequence": "cause"}
# Colonnes catégorielles décrites dans le détail d'un cluster (par exemple les conditions du jour) :
# toutes celles de 2 à CONTEXT_MAX_VALUES valeurs, sauf celles-ci, déjà décrites autrement.
CONTEXT_EXCLUDED = {"site", "asset_type", "severity", "text", "incident_id", "document_format", "extraction_method", "saison"}
CONTEXT_MAX_VALUES = 12
WORD = re.compile(r"(?u)\b[^\W\d_][^\W\d_]+\b")
# Table des phrases (§2.3 du plan) : une ligne par phrase à ranger.
PHRASE_COLUMNS = ["phrase_id", "position", "asset_type", "dimension", "role", "texte", "theme_id"]
ROLES = ("racine", "contributive", "consequence")
COMPONENT_UNSPECIFIED = "non précisé"


# ------------------------------------------------------------------------------------------------
# Petites fonctions de calcul et de vérification
# ------------------------------------------------------------------------------------------------

def percent(series: pd.Series, top: int | None = None) -> dict[str, float]:
    """Répartition en % des valeurs d'une colonne (les plus fréquentes d'abord)."""
    shares = (series.dropna().value_counts(normalize=True) * 100).round(1)
    return (shares.head(top) if top else shares).to_dict()


def keywords_by_cluster(texts: list[str], labels: np.ndarray, stopwords: list[str], n: int = 8) -> dict[int, list[str]]:
    """Mots-clés de chaque cluster (c-TF-IDF, comme BERTopic) : fréquents dans le cluster, rares ailleurs."""
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words=FRENCH_STOPWORDS + stopwords, ngram_range=NGRAM_RANGE,
                                 token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    counts = vectorizer.fit_transform(texts)
    clusters = sorted(set(labels))
    tf = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in clusters]).astype(float)
    idf = np.log(1 + tf.sum(axis=1).mean() / np.maximum(tf.sum(axis=0), 1))
    scores = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1) * idf
    vocab = vectorizer.get_feature_names_out()
    return {c: [vocab[i] for i in scores[k].argsort()[::-1][:n]] for k, c in enumerate(clusters)}


def generic_words(df: pd.DataFrame, segments: dict[str, str]) -> list[str]:
    """Mots qui ne distinguent aucune cause : présents dans au moins GENERIC_MIN_DF des descriptions
    d'un segment (nom de l'installation...), et mots des libellés de segment. Trouvés dans les données."""
    words = set(EXTRA_STOPWORDS)
    for segment, label in segments.items():
        texts = df.loc[df["asset_type"] == segment, "text"].dropna().astype(str).str.lower()
        if len(texts):
            df_counts = pd.Series([w for t in texts for w in set(WORD.findall(t))]).value_counts()
            words |= set(df_counts[df_counts >= GENERIC_MIN_DF * len(texts)].index)
        words |= {w for w in WORD.findall(label.lower()) if len(w) >= 4} | {w + "s" for w in WORD.findall(label.lower())
                                                                             if len(w) >= 4}
    return sorted(words - set(FRENCH_STOPWORDS))


def normalize_phrase(text: str, generic: re.Pattern) -> str:
    """Phrase utilisée pour le clustering : sans les nombres et leurs unités (« arrêt de 3.2 heures »)
    ni les mots génériques (generic_words), qui sinon forment de faux clusters."""
    text = re.sub(r"\d+(?:[.,]\d+)?\s*(?:heures?|h\b|€|euros?|%|jours?|minutes?|min\b)?", " ", str(text))
    text = generic.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,.;:-") or "?"


def contributing_causes(raw) -> list[dict]:
    """Causes contributives d'une cellule (JSON `[{"texte", "type", "indice"}]`) ; liste vide si absente ou illisible."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    return [c for c in items if isinstance(c, dict) and str(c.get("texte", "")).strip()] if isinstance(items, list) else []


def _ids(series: pd.Series) -> pd.Series:
    """Identifiants de thème, valeurs absentes = None (et non NaN)."""
    return series.astype(object).where(series.notna(), None)


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
        self.cadrage = current()
        self.segments = self.cadrage.segments
        self.sites = {a: sorted(self.df.loc[self.df["asset_type"] == a, "site"].unique()) for a in self.segments}
        self.stopwords = generic_words(self.df, {a: self.cadrage.label(a) for a in self.segments})
        self.generic = re.compile(r"\b(?:" + "|".join(map(re.escape, self.stopwords)) + r")\b", re.IGNORECASE)
        # Colonnes de texte (dtype object, ou str depuis pandas 3) : pas les chiffres ni les dates.
        self.context_columns = [c for c in self.df.columns if c not in CONTEXT_EXCLUDED
                                and not pd.api.types.is_numeric_dtype(self.df[c])
                                and not pd.api.types.is_datetime64_any_dtype(self.df[c])
                                and 2 <= self.df[c].nunique() <= CONTEXT_MAX_VALUES]
        self.causes = pd.DataFrame()            # analyses du LLM, index = position dans self.df
        self.phrases = pd.DataFrame(columns=PHRASE_COLUMNS)  # §2.3 : une ligne par phrase, index = label
        self.built: set[str] = set()            # segments dont la table des phrases est construite
        self.vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # "type/dim" -> (labels de phrases, embeddings)
        self.reduced: dict[str, dict[int, np.ndarray]] = {}  # "type/dim" -> UMAP par graine (calculé une fois)
        self.runs: dict[str, dict] = {}         # essais de clustering, par run_id (principal ou reprise)
        self.selected: dict[str, dict] = {}     # "type/dim" -> {"run_id", "justification"} : clustering principal
        self.active: dict[str, str] = {}        # "type/dim" -> run_id sur lequel travaillent les outils de thèmes
        self.themes: dict[str, dict] = {}       # theme_id -> thème
        self.priority_chains: dict[str, dict] = {}  # type -> {"chaines", "justification"} (select_priority_chains)
        self.nominal: dict[str, dict] = {}      # type -> {"theme_ids", "justification"} (flag_nominal)
        self.components: dict[str, dict[str, str]] = {}  # type -> {composant_brut.strip(): groupe} (pipeline)
        self.report: dict | None = None         # contenu du rapport (rédigé ensuite par writing.py)
        self._theme_count = 0

    @property
    def tools(self) -> dict:
        return {"overview": self.overview, "extract_causes": self.extract_causes,
                "run_clustering": self.run_clustering, "select_clustering": self.select_clustering,
                "topic_details": self.topic_details, "define_themes": self.define_themes,
                "define_theme": self.define_theme, "compare_clusters": self.compare_clusters,
                "merge_themes": self.merge_themes,
                "remove_theme": self.remove_theme, "heatmap_hotspots": self.heatmap_hotspots,
                "select_priority_chains": self.select_priority_chains, "nominal_evidence": self.nominal_evidence,
                "flag_nominal": self.flag_nominal}

    # ---- état et vérifications -----------------------------------------------------------------

    @staticmethod
    def key(asset_type: str, dimension: str) -> str:
        return f"{asset_type}/{dimension}"

    def check_args(self, asset_type: str, dimension: str | None = None) -> None:
        check_choice(asset_type, self.segments, "asset_type")
        if dimension is not None:
            check_choice(dimension, list(DIMENSIONS), "dimension")

    def positions(self, asset_type: str) -> np.ndarray:
        return np.flatnonzero(self.df["asset_type"].to_numpy() == asset_type)

    def require_causes(self, asset_type: str) -> None:
        if not set(self.positions(asset_type)) <= set(self.causes.index):
            raise ToolError(f"causes non extraites pour {asset_type} : appeler extract_causes d'abord")
        if asset_type not in self.built:
            raise ToolError(f"phrases non construites pour {asset_type} : extract_causes d'abord")

    def phrase_rows(self, asset_type: str, dimension: str) -> pd.Index:
        """Labels de self.phrases du segment et de la dimension, dans l'ordre de la table."""
        p = self.phrases
        return p.index[(p["asset_type"] == asset_type) & (p["dimension"] == dimension)]

    def phrase_vectors(self, asset_type: str, dimension: str) -> tuple[np.ndarray, np.ndarray]:
        """Labels de self.phrases (toutes les phrases du segment et de la dimension) et leurs embeddings normalisés.
        Causes : racines et contributives ; conséquences : conséquences. Mis en cache par segment et dimension."""
        from topic_modeling.embeddings import embed_texts

        k = self.key(asset_type, dimension)
        if k not in self.vectors:
            self.require_causes(asset_type)
            rows = self.phrase_rows(asset_type, dimension)
            if len(rows) == 0:
                raise ToolError(f"aucune phrase de {dimension} pour {asset_type}")
            emb = embed_texts([normalize_phrase(p, self.generic) for p in self.phrases.loc[rows, "texte"]], CACHE_DIR)
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            self.vectors[k] = (rows.to_numpy(), emb)
        return self.vectors[k]

    def active_vectors(self, asset_type: str, dimension: str) -> tuple[np.ndarray, np.ndarray]:
        """Labels et embeddings des phrases de l'essai actif, dans l'ordre de ses labels. Un essai de reprise ne
        porte que sur les phrases restées sans thème : ses labels de cluster ne sont alignés que sur ce sous-ensemble."""
        run = self.chosen(asset_type, dimension)
        positions, vectors = self.phrase_vectors(asset_type, dimension)
        return run["positions"], vectors[pd.Index(positions).get_indexer(run["positions"])]

    def uncertain_or_unknown(self, positions) -> np.ndarray:
        """Incidents sans cause exploitable : inconnue, ou inférée avec une confiance faible."""
        c = self.causes.loc[positions]
        return ((c["cause_type"] == "inconnue") | ((c["cause_type"] == "inférée") & (c["confiance"] == "faible"))).to_numpy()

    def chosen(self, asset_type: str, dimension: str) -> dict:
        """Essai sur lequel travaillent les outils de thèmes : celui de set_active_run, sinon le clustering choisi."""
        k = self.key(asset_type, dimension)
        if k not in self.selected:
            raise ToolError(f"aucun clustering choisi pour {k} : appeler select_clustering d'abord")
        return self.runs[self.active.get(k) or self.selected[k]["run_id"]]

    def topic_ids(self, asset_type: str, dimension: str) -> list[int]:
        return sorted(int(t) for t in set(self.chosen(asset_type, dimension)["labels"]) if t != -1)

    def run_themes(self, asset_type: str, dimension: str) -> dict[str, dict]:
        """Thèmes créés sur l'essai actif (leur champ run_id)."""
        run_id = self.chosen(asset_type, dimension)["run_id"]
        return {tid: th for tid, th in self.themes.items() if th["asset_type"] == asset_type
                and th["dimension"] == dimension and th.get("run_id") == run_id}

    def theme_of(self, asset_type: str, dimension: str, topic: int) -> str | None:
        return next((tid for tid, th in self.run_themes(asset_type, dimension).items() if topic in th["topics"]), None)

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

    def centroid(self, asset_type: str, dimension: str, mask: np.ndarray) -> np.ndarray:
        _, vectors = self.active_vectors(asset_type, dimension)
        center = vectors[mask].mean(axis=0)
        return center / max(np.linalg.norm(center), 1e-12)

    def component_of(self, asset_type: str, raw) -> str:
        """Composant normalisé d'une valeur brute (pipeline : self.components), « non précisé » sinon."""
        if not isinstance(raw, str) or not raw.strip():
            return COMPONENT_UNSPECIFIED
        return self.components.get(asset_type, {}).get(raw.strip(), COMPONENT_UNSPECIFIED)

    def is_nominal(self, asset_type: str, theme_id, severity) -> bool:
        """Arrêt nominal (§2.4) : thème de conséquence flaggé nominal ET sévérité la plus faible."""
        ids = self.nominal.get(asset_type, {}).get("theme_ids", [])
        return bool(isinstance(theme_id, str) and theme_id in ids and severity == self.cadrage.mildest)

    def incidents_table(self) -> pd.DataFrame:
        """Chaque incident avec son analyse et ses thèmes (plan §2.4). La cause (theme_cause) et la conséquence
        viennent des phrases racine et conséquence ; themes_contributifs_ids des causes contributives vérifiées."""
        out = self.df.join(self.causes)
        names = {tid: th["name"] for tid, th in self.themes.items()}
        phrases = self.phrases
        root = phrases[phrases["role"] == "racine"].set_index("position")["theme_id"]
        consequence = phrases[phrases["role"] == "consequence"].set_index("position")["theme_id"]
        contrib = (phrases[(phrases["role"] == "contributive") & phrases["theme_id"].notna()]
                   .groupby("position")["theme_id"].agg(lambda s: "|".join(dict.fromkeys(s))))
        out["theme_cause_id"] = _ids(root.reindex(out.index))
        out["theme_consequence_id"] = _ids(consequence.reindex(out.index))
        out["themes_contributifs_ids"] = contrib.reindex(out.index).fillna("")
        out["theme_cause"] = out["theme_cause_id"].map(names).fillna(F.UNCLASSIFIED)
        out["theme_consequence"] = out["theme_consequence_id"].map(names).fillna(F.UNCLASSIFIED)
        if "cause_type" in out:
            out.loc[out["cause_type"] == "inconnue", "theme_cause"] = F.UNKNOWN
            out.loc[(out["cause_type"] == "inférée") & (out["confiance"] == "faible"), "theme_cause"] = F.UNCERTAIN
        raw = out["composant_brut"] if "composant_brut" in out else pd.Series("", index=out.index)
        out["composant"] = [self.component_of(a, r) for a, r in zip(out["asset_type"], raw)]
        out["nominal"] = [self.is_nominal(a, t, s) for a, t, s in
                          zip(out["asset_type"], out["theme_consequence_id"], out["severity"])]
        return out

    def unassign(self, theme_ids: list[str]) -> None:
        """Retire ces thèmes des phrases qui les portent (thèmes supprimés ou fusionnés)."""
        self.phrases.loc[self.phrases["theme_id"].isin(theme_ids), "theme_id"] = None

    # ---- table des phrases et affectation (pipeline) ----------------------------------------------

    def build_phrases(self, asset_type: str) -> dict[str, int]:
        """Construit la table des phrases (§2.3) du segment ; appelé par extract_causes, une fois les causes du
        segment connues. Une seule fois par segment, avant tout clustering : refaire le segment après un clustering
        invaliderait ses positions. Causes : la cause racine des incidents dont la cause n'est ni inconnue ni
        incertaine (rôle « racine »), et chaque cause contributive vérifiée (« contributive »). Conséquences : la
        conséquence non vide (« consequence »). Les lignes des autres segments gardent leur label.
        Retourne le nombre de phrases par rôle."""
        self.check_args(asset_type)
        positions = [p for p in self.positions(asset_type) if p in self.causes.index]
        rows = []

        def add(p, role: str, k: int, texte: str) -> None:
            rows.append({"phrase_id": f"{p}:{role}:{k}", "position": int(p), "asset_type": asset_type,
                         "dimension": "consequence" if role == "consequence" else "cause", "role": role,
                         "texte": texte.strip(), "theme_id": None})

        if positions:
            weak = dict(zip(positions, self.uncertain_or_unknown(positions)))
            for p in positions:
                c = self.causes.loc[p]
                if isinstance(c["erreur_analyse"], str) and c["erreur_analyse"]:
                    continue
                racine = c["cause_racine"]
                if not weak[p] and isinstance(racine, str) and racine.strip():
                    add(p, "racine", 0, racine)
                for k, item in enumerate(contributing_causes(c["causes_contributives"])):
                    add(p, "contributive", k, item["texte"])
                if isinstance(c["consequence"], str) and c["consequence"].strip():
                    add(p, "consequence", 0, c["consequence"])
        new = pd.DataFrame(rows, columns=PHRASE_COLUMNS)
        start = int(self.phrases.index.max()) + 1 if len(self.phrases) else 0
        new.index = pd.RangeIndex(start, start + len(new))
        old = self.phrases[self.phrases["asset_type"] != asset_type]
        self.phrases = pd.concat([old, new]).astype({"position": int})
        self.built.add(asset_type)
        for d in DIMENSIONS:  # vecteurs et réductions du segment : à recalculer
            self.vectors.pop(self.key(asset_type, d), None)
            self.reduced.pop(self.key(asset_type, d), None)
        return {role: int((new["role"] == role).sum()) for role in ROLES}

    def theme_cards(self, asset_type: str, dimension: str) -> list[dict]:
        """Fiches des thèmes de l'essai actif, entrée de assign_phrases : id, nom, description et 3 phrases
        typiques (les plus proches du centre du thème)."""
        self.check_args(asset_type, dimension)
        run = self.chosen(asset_type, dimension)
        positions, vectors = self.active_vectors(asset_type, dimension)
        cards = []
        for tid, th in sorted(self.run_themes(asset_type, dimension).items()):
            mask = np.isin(run["labels"], th["topics"])
            center = self.centroid(asset_type, dimension, mask)
            order = np.argsort(-(vectors[mask] @ center))
            cards.append({"id": tid, "name": th["name"], "description": th["description"],
                          "examples": self.phrases.loc[positions[mask][order], "texte"].head(3).tolist()})
        return cards

    def apply_assignment(self, asset_type: str, dimension: str, theme_ids: list[str | None],
                         phrase_index=None) -> dict:
        """Écrit theme_id (None = aucun thème) pour les phrases du segment et de la dimension. Sans phrase_index,
        toutes les phrases de la dimension, dans l'ordre de phrase_vectors ; avec phrase_index (labels de
        self.phrases, par exemple ceux de none_phrases), seulement celles-là. theme_ids est aligné sur ces lignes."""
        self.check_args(asset_type, dimension)
        key_rows = self.phrase_rows(asset_type, dimension)
        target = key_rows if phrase_index is None else pd.Index(list(phrase_index))
        if not target.isin(key_rows).all():
            raise ValueError(f"phrase_index contient des phrases hors de {self.key(asset_type, dimension)}")
        if len(theme_ids) != len(target):
            raise ValueError(f"{len(theme_ids)} affectations pour {len(target)} phrases")
        valid = {tid for tid, th in self.themes.items() if th["asset_type"] == asset_type and th["dimension"] == dimension}
        unknown = [t for t in theme_ids if t is not None and t not in valid]
        if unknown:
            raise ValueError(f"thèmes inconnus pour {self.key(asset_type, dimension)} : {unknown}")
        column = self.phrases["theme_id"].astype(object)
        column.loc[target] = list(theme_ids)
        self.phrases["theme_id"] = column
        return {"phrases": len(target), "aucun": sum(t is None for t in theme_ids)}

    def none_phrases(self, asset_type: str, dimension: str) -> np.ndarray:
        """Labels de self.phrases des phrases du segment et de la dimension sans thème (« aucun »)."""
        rows = self.phrase_rows(asset_type, dimension)
        return rows[self.phrases.loc[rows, "theme_id"].isna().to_numpy()].to_numpy()

    def none_share(self, asset_type: str, dimension: str) -> float:
        """Part (%) des phrases du segment et de la dimension sans thème."""
        total = len(self.phrase_rows(asset_type, dimension))
        return round(100 * len(self.none_phrases(asset_type, dimension)) / total, 1) if total else 0.0

    def set_active_run(self, asset_type: str, dimension: str, run_id: str) -> None:
        """Fait porter les outils de thèmes (topic_details, define_themes, merge_themes, theme_cards...) sur l'essai
        run_id : le clustering principal choisi, ou une reprise de recluster_none. Ne supprime aucun thème : les thèmes
        créés sur un essai portent son run_id et restent affectés à leurs phrases. Pour revenir au principal, passer
        l'id de selected[key]["run_id"]."""
        self.check_args(asset_type, dimension)
        k = self.key(asset_type, dimension)
        if run_id not in self.runs or self.runs[run_id]["base"] != k:
            raise ValueError(f"essai inconnu pour {k} : {run_id}")
        self.active[k] = run_id

    def recluster_none(self, asset_type: str, dimension: str) -> str | None:
        """Reprise du clustering sur les phrases sans thème seulement : une passe HDBSCAN
        (min_cluster_size = RECLUSTER_MIN_CLUSTER_SIZE, un reste homogène peut former un seul cluster,
        UMAP à la graine STABILITY_SEEDS[0] si assez de phrases).
        L'essai est enregistré dans self.runs sous run_id f"{asset_type}_{dimension}_reprise{n}" (n = 1, 2...),
        avec key = clé du segment + "/reprise{n}" et base = clé du segment : il n'entre ni dans le budget des essais
        ni dans select_clustering. Pour le thématiser, le pipeline appelle, dans l'ordre :
            run_id = recluster_none(a, d)                 # None si pas assez de phrases ou aucun cluster
            set_active_run(a, d, run_id)                  # les outils de thèmes travaillent sur la reprise
            cartes = theme_cards(a, d)                    # thèmes de la reprise, avec leurs phrases typiques
            ... define_themes / assign_phrases ...        # thèmes créés : run_id = celui de la reprise
            apply_assignment(a, d, ids, phrase_index=none_phrases(a, d))   # phrases restées sans thème seulement
            set_active_run(a, d, selected[key]["run_id"]) # rendre le clustering principal actif
        Retourne le run_id, ou None."""
        from hdbscan import HDBSCAN
        from umap import UMAP

        self.check_args(asset_type, dimension)
        k = self.key(asset_type, dimension)
        none = self.none_phrases(asset_type, dimension)
        if len(none) < 2 * RECLUSTER_MIN_CLUSTER_SIZE:
            return None
        positions, vectors = self.phrase_vectors(asset_type, dimension)
        keep = np.isin(positions, none)
        sub_pos, sub_vec = positions[keep], vectors[keep]
        if len(sub_pos) > UMAP_PARAMS["n_neighbors"]:  # UMAP exige plus de points que de voisins
            sub_vec = UMAP(**{**UMAP_PARAMS, "random_state": STABILITY_SEEDS[0]}).fit_transform(sub_vec)
        # allow_single_cluster : un reste homogène (une seule masse de phrases semblables) doit pouvoir former un cluster
        labels = HDBSCAN(min_cluster_size=RECLUSTER_MIN_CLUSTER_SIZE, cluster_selection_method="eom",
                         allow_single_cluster=True).fit(sub_vec).labels_
        if not (labels != -1).any():
            return None
        n = 1 + sum(1 for r in self.runs.values() if r["base"] == k and r["key"] != k)
        run_id, rkey = f"{asset_type}_{dimension}_reprise{n}", f"{k}/reprise{n}"
        texts = [normalize_phrase(p, self.generic) for p in self.phrases.loc[sub_pos, "texte"]]
        sizes = pd.Series(labels[labels != -1]).value_counts()
        keywords = keywords_by_cluster(texts, labels, self.stopwords)
        summary = {"run_id": run_id, "asset_type": asset_type, "dimension": dimension, "reprise": True,
                   "min_cluster_size": RECLUSTER_MIN_CLUSTER_SIZE, "phrases_reprises": len(sub_pos),
                   "nombre_clusters": len(sizes), "non_classes_pct": round(100 * float((labels == -1).mean()), 1),
                   "clusters": [{"cluster": int(t), "phrases": int(c), "mots_cles": keywords[t][:5]}
                                for t, c in sizes.items()]}
        self.runs[run_id] = {"run_id": run_id, "key": rkey, "base": k, "asset_type": asset_type, "dimension": dimension,
                             "positions": sub_pos, "labels": labels, "keywords": keywords, "summary": summary}
        return run_id

    # ---- outils ----------------------------------------------------------------------------------

    def overview(self) -> dict:
        sites = (self.df.groupby(["asset_type", "site"])
                 .agg(incidents=("incident_id", "count"), cout_moyen_eur=("estimated_cost_eur", "mean"),
                      arret_moyen_h=("downtime_hours", "mean"))
                 .round(1).reset_index().to_dict(orient="records"))
        return {"incidents": len(self.df), "periode": [self.df["date"].min(), self.df["date"].max()],
                "sites": sites, "severite_pct": percent(self.df["severity"]),
                "segments": {a: self.cadrage.label(a) for a in self.segments},
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
        phrases = self.build_phrases(asset_type)
        sub = self.df.loc[self.positions(asset_type)].join(self.causes)
        ok = sub[sub["erreur_analyse"] == ""]
        examples = ok.sample(min(3, len(ok)), random_state=0)
        contrib = ok["causes_contributives"].map(lambda s: len(contributing_causes(s)))
        return {"incidents_analyses": len(ok), "echecs": int(len(sub) - len(ok)),
                "cause_non_precisee_pct": round(100 * float((ok["cause_racine"] == CAUSE_INCONNUE).mean()), 1)
                if len(ok) else 0.0,
                "type_de_cause_pct": percent(ok["cause_type"]),
                "causes_contributives": {
                    "moyenne_verifiees_par_incident": round(float(contrib.mean()), 2) if len(ok) else 0.0,
                    "rejetees_total": int(pd.to_numeric(ok["contributives_rejetees"], errors="coerce").fillna(0).sum())},
                "phrases": phrases,
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
        phrases = [normalize_phrase(p, self.generic) for p in self.phrases.loc[positions, "texte"]]
        keywords = keywords_by_cluster(phrases, labels, self.stopwords)

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
            "clusters": [{"cluster": int(t), "phrases": int(n), "mots_cles": keywords[t][:5]} for t, n in sizes.items()],
        }
        summary["qualite"] = F.clustering_quality(summary)
        self.runs[run_id] = {"run_id": run_id, "key": k, "base": k, "asset_type": asset_type, "dimension": dimension,
                             "positions": positions, "labels": labels, "keywords": keywords, "summary": summary}
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
        self.unassign(removed)
        self.selected[k] = {"run_id": run_id, "justification": justification}
        self.active[k] = run_id
        return {"run_id": run_id, "clusters_a_ranger": self.topic_ids(asset_type, dimension), "themes_supprimes": removed}

    def topic_details(self, asset_type: str, dimension: str, topics: list[int]) -> dict:
        self.check_args(asset_type, dimension)
        run = self.chosen(asset_type, dimension)
        known = self.topic_ids(asset_type, dimension)
        topics = check_list(topics, MAX_TOPICS_PER_CALL, "topics")
        for t in topics:
            check_choice(t, known, "topic")
        positions, vectors = self.active_vectors(asset_type, dimension)
        other = OTHER[dimension]
        details = []
        for t in topics:
            mask = run["labels"] == t
            center = vectors[mask].mean(axis=0)
            typical = self.phrases.loc[positions[mask][np.argsort(-(vectors[mask] @ center))]]  # plus proches d'abord
            incidents = list(dict.fromkeys(typical["position"]))  # incidents des phrases, les plus typiques d'abord
            sub = self.df.loc[incidents].join(self.causes)
            autre = self.phrases[self.phrases["position"].isin(incidents) & (self.phrases["dimension"] == other)
                                 & (self.phrases["role"] != "contributive")]
            detail = {
                "cluster": t, "phrases": len(typical), "incidents": len(incidents), "mots_cles": run["keywords"][t],
                "cohesion": round(self.cohesion(vectors[mask]), 3), "theme": self.theme_of(asset_type, dimension, t),
                "phrases_typiques": typical["texte"].head(N_EXAMPLES).tolist(),
                f"{other}_exemples": autre["texte"].head(2).tolist(),
                "description_exemple": [str(self.df.loc[p, "text"])[:EXAMPLE_MAX_CHARS] for p in incidents[:N_DESCRIPTIONS]],
                "sites_pct": percent(sub["site"]), "severite_pct": percent(sub["severity"]),
                "saisons_pct": percent(sub["saison"], top=2),
                **{f"{c}_pct": percent(sub[c], top=3) for c in self.context_columns},
                "cout_moyen_eur": round(float(sub["estimated_cost_eur"].mean())),
            }
            if dimension == "cause":
                detail["type_de_cause_pct"] = percent(sub["cause_type"])
            details.append(detail)
        return {"clusters": details, "restants_sans_theme": self.unthemed(asset_type, dimension)}

    def define_themes(self, asset_type: str, dimension: str, themes: list[dict]) -> dict:
        """Crée plusieurs thèmes ; chacun est accepté ou refusé indépendamment."""
        self.check_args(asset_type, dimension)
        themes = check_list(themes, MAX_THEMES_PER_CALL, "themes")
        known = self.topic_ids(asset_type, dimension)
        _, vectors = self.active_vectors(asset_type, dimension)
        run = self.chosen(asset_type, dimension)
        labels = run["labels"]
        results = []
        for theme in themes:
            try:
                if not isinstance(theme, dict):
                    raise ToolError("chaque thème doit être un objet {name, topics, description, justification}")
                name = check_text(theme.get("name"), "name", MAX_THEME_NAME_CHARS)
                if dimension == "cause" and re.search(CONSEQUENCE_WORDS, name, re.IGNORECASE):
                    raise ToolError(f"nom refusé : « {name} » décrit une conséquence (arrêt, perte, dégât, feu...). Un "
                                    "thème de cause ne nomme que ce qui a PROVOQUÉ l'incident.")
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
                if len(topics) > 1:  # fusion de clusters : les phrases réunies doivent rester semblables
                    reference = np.average([self.cohesion(vectors[labels == t]) for t in topics],
                                           weights=[(labels == t).sum() for t in topics])
                    if cohesion < MERGE_MIN_COHESION_RATIO * reference:
                        raise ToolError(f"fusion refusée : cohésion {cohesion:.3f} contre {reference:.3f} dans les "
                                        f"clusters d'origine (minimum {MERGE_MIN_COHESION_RATIO:.0%}) : trop différents")
                self._theme_count += 1
                theme_id = f"{asset_type}_{dimension}_T{self._theme_count:02d}"
                self.themes[theme_id] = {"asset_type": asset_type, "dimension": dimension, "name": name,
                                         "description": description, "topics": topics, "justification": justification,
                                         "run_id": run["run_id"], "phrases": int(mask.sum()),
                                         "cohesion": round(cohesion, 3)}
                results.append({"name": name, "ok": True, "theme_id": theme_id, "phrases": int(mask.sum()),
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

    def similar_cluster_pairs(self, asset_type: str, dimension: str) -> list[tuple[int, int, float]]:
        """Paires de clusters à fusionner : centres dont la similarité atteint MERGE_SIMILARITY."""
        labels = self.chosen(asset_type, dimension)["labels"]
        topics = self.topic_ids(asset_type, dimension)
        centers = {t: self.centroid(asset_type, dimension, labels == t) for t in topics}
        pairs = [(a, b, round(float(centers[a] @ centers[b]), 3)) for i, a in enumerate(topics) for b in topics[i + 1:]]
        return [p for p in pairs if p[2] >= MERGE_SIMILARITY]

    def duplicate_themes(self, asset_type: str, dimension: str) -> list[tuple[str, str, float]]:
        """Thèmes de l'essai actif qui décrivent la même chose (centres de similarité >= MERGE_SIMILARITY) : à fusionner."""
        labels = self.chosen(asset_type, dimension)["labels"]
        themes = self.run_themes(asset_type, dimension)
        centers = {tid: self.centroid(asset_type, dimension, np.isin(labels, th["topics"])) for tid, th in themes.items()}
        ids = sorted(themes)
        return [(a, b, round(float(centers[a] @ centers[b]), 3)) for i, a in enumerate(ids) for b in ids[i + 1:]
                if float(centers[a] @ centers[b]) >= MERGE_SIMILARITY]

    def compare_clusters(self, asset_type: str, dimension: str, a: int, b: int) -> dict:
        """Les clusters a et b décrivent-ils la même chose ? Similarité de leurs centres, exemples et avis."""
        self.check_args(asset_type, dimension)
        known = self.topic_ids(asset_type, dimension)
        check_choice(a, known, "a")
        check_choice(b, known, "b")
        labels = self.chosen(asset_type, dimension)["labels"]
        positions, _ = self.active_vectors(asset_type, dimension)
        sim = float(self.centroid(asset_type, dimension, labels == a) @ self.centroid(asset_type, dimension, labels == b))
        advice = "fusion recommandée" if sim >= MERGE_SIMILARITY else "fusion déconseillée : centres trop éloignés"
        return {"similarite": round(sim, 3), "seuil": MERGE_SIMILARITY, "avis": advice,
                "exemples": {t: self.phrases.loc[positions[labels == t], "texte"].head(3).tolist() for t in (a, b)}}

    def merge_themes(self, asset_type: str, dimension: str, theme_ids: list[str], name: str) -> dict:
        """Fusionne des thèmes déjà créés en un seul (pour supprimer un doublon)."""
        self.check_args(asset_type, dimension)
        theme_ids = check_list(theme_ids, 10, "theme_ids")
        own = list(self.run_themes(asset_type, dimension))
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
        self.unassign(theme_ids)
        return {**outcome, "doublons_restants": self.duplicate_themes(asset_type, dimension)}

    def remove_theme(self, theme_id: str) -> dict:
        check_choice(theme_id, list(self.themes), "theme_id")
        theme = self.themes.pop(theme_id)
        self.unassign([theme_id])
        return {"clusters_liberes": theme["topics"],
                "restants_sans_theme": self.unthemed(theme["asset_type"], theme["dimension"])}

    def heatmap_hotspots(self, asset_type: str, site: str | None = None, top_k: int = HOTSPOTS_MAX) -> dict:
        """Cases les plus coûteuses de la heatmap causes x conséquences (calculées en Python, facts.py)."""
        self.check_args(asset_type)
        if site is not None:
            check_choice(site, self.sites[asset_type], "site")
        top_k = check_int(top_k, 1, HOTSPOTS_MAX, "top_k")
        self.require_themes(asset_type)
        table = self.incidents_table()
        table = table[table["asset_type"] == asset_type]
        if site is not None:
            table = table[table["site"] == site]
        return {"asset_type": asset_type, "site": site or "tous", "incidents": len(table),
                "cout_total_perimetre": round(float(table["estimated_cost_eur"].sum())),
                "cases": F.hotspots(table, top_k),
                "lecture": "part_cout_pct = part du coût total du périmètre ; arrêts de sécurité nominaux exclus"}

    def select_priority_chains(self, asset_type: str, chains: list[str], justification: str) -> dict:
        """Outil final de la phase priorités : les chaînes doivent être parmi les HOTSPOTS_MAX cases les
        plus coûteuses du type, et la justification ne citer que des chiffres de ces cases."""
        from .writing import cause_names, concept_patterns, text_errors

        self.check_args(asset_type)
        self.require_themes(asset_type)
        justification = check_text(justification, "justification", 1500)
        low, high = PRIORITY_CHAINS_RANGE
        table = self.incidents_table()
        top = F.hotspots(table[table["asset_type"] == asset_type], HOTSPOTS_MAX)
        allowed = {c["id"]: c for c in top}
        chains = list(dict.fromkeys(check_list(chains, high, "chains")))
        if len(chains) < min(low, len(allowed)):
            raise ToolError(f"choisis au moins {min(low, len(allowed))} chaînes (reçu : {len(chains)})")
        for c in chains:
            check_choice(c, list(allowed), "chaîne")
        facts = "\n".join(str(c) for c in top) + f"\n{len(table[table['asset_type'] == asset_type])} incidents"
        segment = table[table["asset_type"] == asset_type]
        errors = text_errors(justification, facts, concept_patterns(cause_names(segment)))
        if errors:
            raise ToolError("justification refusée : " + " ; ".join(errors))
        self.priority_chains[asset_type] = {"chaines": [allowed[c] for c in chains], "justification": justification}
        return {"message": "chaînes prioritaires enregistrées", "chaines": chains}

    def consequence_evidence(self, asset_type: str) -> dict[str, dict]:
        """Par thème de conséquence du segment (tous essais) : incidents distincts, part à la sévérité la plus
        faible, coût moyen, 3 exemples. Base de nominal_evidence et de flag_nominal."""
        themes = {tid: th for tid, th in self.themes.items()
                  if th["asset_type"] == asset_type and th["dimension"] == "consequence"}
        table = self.incidents_table()
        table = table[table["asset_type"] == asset_type]
        p = self.phrases
        mine = p[(p["asset_type"] == asset_type) & (p["role"] == "consequence") & p["theme_id"].notna()]
        mildest = self.cadrage.mildest
        out = {}
        for tid, th in themes.items():
            phrases = mine[mine["theme_id"] == tid]
            inc = table.loc[phrases["position"].unique()]
            n = len(inc)
            out[tid] = {"theme_id": tid, "name": th["name"], "incidents": n,
                        "pct_severite_la_plus_faible": round(100 * float((inc["severity"] == mildest).mean()), 1) if n else 0.0,
                        "cout_moyen_eur": round(float(inc["estimated_cost_eur"].mean())) if n else 0,
                        "exemples": phrases["texte"].head(3).tolist()}
        return out

    def nominal_evidence(self, asset_type: str) -> dict:
        """Pour chaque thème de conséquence du segment : incidents, part à la sévérité la plus faible, coût moyen,
        exemples. Sert à choisir les arrêts nominaux (flag_nominal)."""
        self.check_args(asset_type)
        return {"asset_type": asset_type, "severite_la_plus_faible": self.cadrage.mildest,
                "seuil_pct": NOMINAL_MIN_MILDEST_PCT,
                "lecture": f"Un thème n'est nominal que si au moins {NOMINAL_MIN_MILDEST_PCT:g} % de ses incidents "
                           f"sont à la sévérité la plus faible.",
                "themes": list(self.consequence_evidence(asset_type).values())}

    def flag_nominal(self, asset_type: str, theme_ids: list[str], justification: str) -> dict:
        """Outil final de la phase nominale : thèmes de conséquences qui sont des arrêts nominaux. Liste vide permise.
        Refusé pour tout thème dont la part à la sévérité la plus faible est sous NOMINAL_MIN_MILDEST_PCT."""
        self.check_args(asset_type)
        justification = check_text(justification, "justification", 1500)
        if not isinstance(theme_ids, list) or not all(isinstance(t, str) for t in theme_ids):
            raise ToolError("theme_ids doit être une liste d'identifiants de thèmes (liste vide permise)")
        theme_ids = list(dict.fromkeys(theme_ids))
        evidence = self.consequence_evidence(asset_type)
        for tid in theme_ids:
            check_choice(tid, list(evidence), "theme_id")
        refused = [f"{tid} ({evidence[tid]['name']}) : {evidence[tid]['pct_severite_la_plus_faible']} % des incidents "
                   f"à la sévérité la plus faible, {NOMINAL_MIN_MILDEST_PCT:g} % exigés"
                   for tid in theme_ids if evidence[tid]["pct_severite_la_plus_faible"] < NOMINAL_MIN_MILDEST_PCT]
        if refused:
            raise ToolError("arrêt nominal refusé : " + " ; ".join(refused) + ". Retire ces thèmes.")
        self.nominal[asset_type] = {"theme_ids": theme_ids, "justification": justification}
        return {"message": "arrêts nominaux enregistrés", "theme_ids": theme_ids}


DIMENSION = {"type": "string", "enum": list(DIMENSIONS), "description": "cause (pourquoi) ou consequence (ce que "
                                                                        "l'incident a provoqué)"}


def tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or list(properties),
                       "additionalProperties": False}}}


def tool_schemas() -> list[dict]:
    """Schémas des outils : les segments viennent du cadrage."""
    c = current()
    ASSET = {"type": "string", "enum": c.segments, "description": "segment : " + ", ".join(
        f"{a} ({c.label(a)})" for a in c.segments)}
    return [
        tool("overview", "Vue d'ensemble : incidents par site, sévérité, segments, bornes et seuils.", {}),
        tool("extract_causes", "Fait extraire par le LLM, pour chaque incident du segment, la cause racine (pourquoi), "
                               "les causes contributives (conditions) et la conséquence (ce qui a été provoqué).",
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
                              "sites, sévérité, saisons, contexte, coût, cohésion (1 = phrases identiques).",
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
                                 "des exemples de phrases et un avis de fusion.",
             {"asset_type": ASSET, "dimension": DIMENSION, "a": {"type": "integer"}, "b": {"type": "integer"}}),
        tool("merge_themes", "Fusionne plusieurs thèmes déjà créés en un seul (pour supprimer un doublon).",
             {"asset_type": ASSET, "dimension": DIMENSION, "theme_ids": {"type": "array", "items": {"type": "string"}},
              "name": {"type": "string"}}),
        tool("remove_theme", "Supprime un thème pour libérer ses clusters.", {"theme_id": {"type": "string"}}),
        tool("heatmap_hotspots", f"Cases les plus coûteuses de la heatmap thèmes de causes connues x thèmes de conséquences "
                                 f"(1 à {HOTSPOTS_MAX}) : id, cause, conséquence, incidents, coût total et moyen, arrêt, part "
                                 "du coût, part de maintenance suspectée. Sur tout le type, ou sur un site.",
             {"asset_type": ASSET, "site": {"type": ["string", "null"], "description": "un site du type, ou null pour tous"},
              "top_k": {"type": "integer", "minimum": 1, "maximum": HOTSPOTS_MAX}}, required=["asset_type"]),
        tool("select_priority_chains", f"Outil final : choisit {PRIORITY_CHAINS_RANGE[0]} à {PRIORITY_CHAINS_RANGE[1]} chaînes "
                                       "cause -> conséquence prioritaires (id de heatmap_hotspots sans site).",
             {"asset_type": ASSET, "chains": {"type": "array", "items": {"type": "string"}, "maxItems": PRIORITY_CHAINS_RANGE[1]},
              "justification": {"type": "string", "description": "pourquoi ces chaînes, avec les chiffres de l'outil"}}),
        tool("nominal_evidence", "Pour chaque thème de conséquence du segment : nombre d'incidents, part à la sévérité la "
                                 "plus faible, coût moyen et exemples. Sert à choisir les arrêts nominaux.",
             {"asset_type": ASSET}),
        tool("flag_nominal", f"Outil final : choisit les thèmes de conséquences qui sont des arrêts nominaux (une liste vide "
                             f"est permise). Un thème n'est accepté que si au moins {NOMINAL_MIN_MILDEST_PCT:g} % de ses "
                             "incidents sont à la sévérité la plus faible.",
             {"asset_type": ASSET, "theme_ids": {"type": "array", "items": {"type": "string"}},
              "justification": {"type": "string"}}),
    ]
