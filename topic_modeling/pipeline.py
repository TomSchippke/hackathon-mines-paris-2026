"""Topic modeling BERTopic sur tous les incidents (texte, PDF digitaux et scans).

Étapes :
  1. Lecture de `outputs/1_extraction/incidents_full.csv` (sortie de pdf_pipeline).
  2. Embeddings Azure OpenAI des descriptions (cache : seuls les nouveaux textes sont envoyés).
  3. Un modèle BERTopic par type d'installation (PV / éolien) : UMAP -> HDBSCAN -> mots-clés.
  4. Sorties dans outputs/2_topics/ :
       incidents_topics.csv   chaque incident avec son topic ;
       topic_info.csv         chaque topic : taille, mots-clés, exemples représentatifs ;
       topics_par_site.csv    nombre d'incidents par topic et par site ;
       barchart_<type>.html   mots-clés des topics (graphique interactif).

Usage :
    python -m topic_modeling
    python -m topic_modeling --min-cluster-size 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths import CACHE_DIR, INCIDENTS_FULL, TOPICS_DIR

from .config import FRENCH_STOPWORDS, MIN_CLUSTER_SIZE, NGRAM_RANGE, SPLIT_COLUMN, UMAP_PARAMS
from .embeddings import embed_texts


def build_model(min_cluster_size: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    return BERTopic(
        umap_model=UMAP(**UMAP_PARAMS),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=FRENCH_STOPWORDS, ngram_range=NGRAM_RANGE,
                                         token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b"),
        language="french",
        calculate_probabilities=True,
    )


def fit_group(name: str, group: pd.DataFrame, embeddings, min_cluster_size: int,
              out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BERTopic sur un groupe d'incidents. Les topics sont préfixés par le groupe ('solar_3') pour
    rester uniques une fois les groupes réunis ; -1 = incident non classé."""
    model = build_model(min_cluster_size)
    topics, probs = model.fit_transform(group["text"].tolist(), embeddings=embeddings)

    info = model.get_topic_info()
    info.insert(0, "group", name)
    info["topic_id"] = [f"{name}_{t}" for t in info["Topic"]]
    info["keywords"] = [", ".join(w for w, _ in model.get_topic(t)[:8]) for t in info["Topic"]]
    info["representative_docs"] = info["Representative_Docs"].map(lambda docs: " ||| ".join(docs))
    info = info.rename(columns={"Topic": "topic", "Count": "count"})[
        ["group", "topic", "topic_id", "count", "keywords", "representative_docs"]]

    assigned = pd.DataFrame({
        "incident_id": group["incident_id"].values,
        "topic": topics,
        "topic_id": [f"{name}_{t}" for t in topics],
        "topic_probability": probs.max(axis=1).round(3) if probs is not None and probs.ndim == 2 else None,
    })
    try:
        model.visualize_barchart(top_n_topics=20, n_words=8).write_html(out_dir / f"barchart_{name}.html")
    except Exception as e:  # graphique facultatif (plotly absent, trop peu de topics...)
        print(f"  graphique {name} non généré : {e}")
    n_topics = (info["topic"] != -1).sum()
    print(f"  {name} : {len(group)} incidents, {n_topics} topics, "
          f"{(assigned['topic'] == -1).sum()} non classés")
    return assigned, info


def run(input_csv: Path, min_cluster_size: int, split: bool) -> Path:
    if not input_csv.exists():
        raise SystemExit(f"{input_csv} introuvable : lancer d'abord l'extraction des PDF "
                         "(python main.py ou python -m pdf_pipeline).")
    df = pd.read_csv(input_csv)
    df = df[df["text"].notna()].reset_index(drop=True)
    print(f"{len(df)} incidents avec une description")

    embeddings = embed_texts(df["text"].tolist(), CACHE_DIR)

    topics_dir = TOPICS_DIR
    topics_dir.mkdir(parents=True, exist_ok=True)
    groups = df.groupby(SPLIT_COLUMN).groups if split else {"all": df.index}
    print(f"BERTopic ({'un modèle par ' + SPLIT_COLUMN if split else 'un seul modèle'}, "
          f"min_cluster_size={min_cluster_size})")
    assigned, infos = [], []
    for name, idx in groups.items():
        a, i = fit_group(str(name), df.loc[idx], embeddings[idx], min_cluster_size, topics_dir)
        assigned.append(a)
        infos.append(i)

    info = pd.concat(infos, ignore_index=True)
    labels = dict(zip(info["topic_id"], info["topic_id"] + " : " + info["keywords"].str.split(", ").str[:4].str.join(", ")))
    result = df.merge(pd.concat(assigned), on="incident_id")
    result["topic_label"] = result["topic_id"].map(labels)
    result.to_csv(topics_dir / "incidents_topics.csv", index=False)
    info.to_csv(topics_dir / "topic_info.csv", index=False)

    by_site = pd.crosstab(result["topic_label"], result["site"])
    by_site["total"] = by_site.sum(axis=1)
    by_site.sort_values("total", ascending=False).to_csv(topics_dir / "topics_par_site.csv")

    print(f"\nSauvegardé dans {topics_dir}/ : incidents_topics.csv, topic_info.csv, topics_par_site.csv")
    top = by_site.drop(columns="total")
    print("\nTopic dominant par site (hors non classés) :")
    for site in top.columns:
        col = top.loc[~top.index.str.contains(r"_-1 :"), site]
        print(f"  {site:22s} {col.idxmax()} ({col.max()} / {top[site].sum()} incidents)")
    return topics_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INCIDENTS_FULL)
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--no-split", action="store_true", help="un seul modèle pour PV et éolien")
    args = parser.parse_args()
    run(args.input, args.min_cluster_size, split=not args.no_split)
