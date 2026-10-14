"""Pipeline agentique de segmentation, diagnostic de causes et évaluation de criticité."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .agent import diagnose_cluster, reassign_outliers_batch
from .clustering import (
    cluster_texts,
    extract_ctfidf_keywords,
    get_cluster_metadata_profile,
    select_representative_documents,
)
from .config import (
    COST_COL,
    DATE_COL,
    DEFAULT_INPUT,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLIT_COL,
    DOWNTIME_COL,
    FALLBACK_INPUT,
    ID_COL,
    MIN_CLUSTER_SIZE,
    OUTLIER_BATCH_SIZE,
    TEXT_COL,
)
from .criticality import compute_cluster_criticality


def validate_and_detect_columns(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    cost_col: str = COST_COL,
    downtime_col: str = DOWNTIME_COL,
    date_col: str = DATE_COL,
    id_col: str = ID_COL,
) -> dict[str, str]:
    """Valide les colonnes requises avec détection intelligente de secours."""
    detected = {}

    # 1. Colonne texte
    if text_col in df.columns:
        detected["text"] = text_col
    else:
        text_candidates = [c for c in df.columns if any(k in c.lower() for k in ["text", "desc", "rapport", "detail", "summary"])]
        if text_candidates:
            detected["text"] = text_candidates[0]
            print(f"[Auto-détection] Colonne texte identifiée : '{detected['text']}'")
        else:
            raise ValueError(f"Impossible de trouver une colonne texte dans le dataset. Colonnes : {list(df.columns)}")

    # 2. Colonne coût
    if cost_col in df.columns:
        detected["cost"] = cost_col
    else:
        cost_candidates = [c for c in df.columns if any(k in c.lower() for k in ["cost", "cout", "coût", "eur", "montant", "price"])]
        if cost_candidates:
            detected["cost"] = cost_candidates[0]
            print(f"[Auto-détection] Colonne coût identifiée : '{detected['cost']}'")
        else:
            raise ValueError(f"Impossible de trouver une colonne coût dans le dataset. Colonnes : {list(df.columns)}")

    # 3. Colonne durée / downtime
    if downtime_col in df.columns:
        detected["downtime"] = downtime_col
    else:
        downtime_candidates = [c for c in df.columns if any(k in c.lower() for k in ["downtime", "duree", "durée", "hours", "heure"])]
        detected["downtime"] = downtime_candidates[0] if downtime_candidates else ""
        if detected["downtime"]:
            print(f"[Auto-détection] Colonne durée identifiée : '{detected['downtime']}'")

    # 4. Colonne date
    if date_col in df.columns:
        detected["date"] = date_col
    else:
        date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "timestamp", "jour"])]
        detected["date"] = date_candidates[0] if date_candidates else ""
        if detected["date"]:
            print(f"[Auto-détection] Colonne date identifiée : '{detected['date']}'")

    # 5. Colonne identifiant
    if id_col in df.columns:
        detected["id"] = id_col
    else:
        id_candidates = [c for c in df.columns if any(k in c.lower() for k in ["id", "num", "ref"])]
        detected["id"] = id_candidates[0] if id_candidates else ""

    return detected


def run_agentic_segmentation(
    input_csv: Path | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    split_col: str | None = DEFAULT_SPLIT_COL,
) -> tuple[Path, Path]:
    """Exécute l'ensemble du pipeline agentique."""
    if input_csv is None:
        if DEFAULT_INPUT.exists():
            input_csv = DEFAULT_INPUT
        elif FALLBACK_INPUT.exists():
            input_csv = FALLBACK_INPUT
        else:
            raise FileNotFoundError(
                f"Fichier d'entrée introuvable ({DEFAULT_INPUT} ou {FALLBACK_INPUT})."
            )

    print(f"\n[1/6] Chargement des incidents depuis {input_csv}...")
    df = pd.read_csv(input_csv)
    cols = validate_and_detect_columns(df)

    if not cols["id"]:
        df["incident_id"] = np.arange(1, len(df) + 1)
        cols["id"] = "incident_id"

    # Filtrage des lignes avec texte valide
    df = df[df[cols["text"]].notna() & (df[cols["text"]].astype(str).str.strip() != "")].copy()
    df[cols["cost"]] = pd.to_numeric(df[cols["cost"]], errors="coerce").fillna(0.0)
    if cols["downtime"]:
        df[cols["downtime"]] = pd.to_numeric(df[cols["downtime"]], errors="coerce").fillna(0.0)

    print(f"       -> {len(df)} incidents exploitables chargés.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Découpage par domaine si disponible
    do_split = split_col and split_col in df.columns
    groups = df.groupby(split_col).groups if do_split else {"global": df.index}

    print(f"\n[2/6] Passe 1 : Clustering UMAP + HDBSCAN ({'partition par ' + split_col if do_split else 'global'})...")

    all_assigned = []
    cluster_profiles = {}  # cluster_id -> dict(keywords, reps, profile)

    for group_name, idx_list in groups.items():
        sub_df = df.loc[idx_list].reset_index(drop=True)
        print(f"       -> Clustering du groupe '{group_name}' ({len(sub_df)} incidents)...")
        labels, embeddings, keywords, reps = cluster_texts(
            sub_df,
            text_col=cols["text"],
            min_cluster_size=min_cluster_size,
        )

        for i, row in sub_df.iterrows():
            lbl = labels[i]
            c_id = f"{group_name}_{lbl}" if lbl != -1 else f"{group_name}_-1"
            all_assigned.append({
                cols["id"]: row[cols["id"]],
                "raw_cluster_id": c_id,
                "is_outlier": (lbl == -1),
                "text": row[cols["text"]],
            })

        # Mémorisation des métadonnées des clusters denses (hors -1)
        sub_df["raw_cluster"] = labels
        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            c_id = f"{group_name}_{lbl}"
            cluster_profiles[c_id] = {
                "cluster_id": c_id,
                "keywords": keywords.get(lbl, []),
                "representative_texts": reps.get(lbl, []),
                "metadata_profile": get_cluster_metadata_profile(sub_df, lbl),
            }

    assigned_df = pd.DataFrame(all_assigned)
    dense_count = (assigned_df["is_outlier"] == False).sum()
    outlier_count = (assigned_df["is_outlier"] == True).sum()
    print(f"       -> Clusters denses formés : {len(cluster_profiles)} groupes ({dense_count} incidents).")
    print(f"       -> Incidents isolés (outliers -1) : {outlier_count} incidents.")

    print(f"\n[3/6] Diagnostic LLM des {len(cluster_profiles)} clusters denses...")
    clusters_diagnoses: dict[str, dict[str, str]] = {}
    for c_id, p in tqdm(cluster_profiles.items(), desc="Diagnostic LLM"):
        diag = diagnose_cluster(
            cluster_id=c_id,
            keywords=p["keywords"],
            representative_texts=p["representative_texts"],
            metadata_profile=p["metadata_profile"],
        )
        clusters_diagnoses[c_id] = diag

    # Taxonomie établie pour la passe 2
    taxonomy_list = [
        {
            "cluster_id": c_id,
            "problem_title": d["problem_title"],
            "common_cause": d["common_cause"],
        }
        for c_id, d in clusters_diagnoses.items()
    ]

    print(f"\n[4/6] Passe 2 : Reclassement LLM des {outlier_count} incidents isolés...")
    outliers_to_reassign = assigned_df[assigned_df["is_outlier"]].copy()

    reassigned_map: dict[Any, dict[str, Any]] = {}
    if not outliers_to_reassign.empty and taxonomy_list:
        outlier_items = outliers_to_reassign[[cols["id"], "text"]].to_dict(orient="records")
        for i in range(0, len(outlier_items), OUTLIER_BATCH_SIZE):
            batch = outlier_items[i : i + OUTLIER_BATCH_SIZE]
            reassigned_batch = reassign_outliers_batch(batch, taxonomy_list)
            for r in reassigned_batch:
                reassigned_map[r["incident_id"]] = r

    # Mise à jour des clusters finaux et sources d'affectation
    final_clusters = []
    assignment_sources = []
    for _, row in assigned_df.iterrows():
        inc_id = row[cols["id"]]
        if not row["is_outlier"]:
            final_clusters.append(row["raw_cluster_id"])
            assignment_sources.append("dense_cluster")
        else:
            reassign_info = reassigned_map.get(inc_id, {})
            assigned_c = reassign_info.get("assigned_cluster_id", "isole_atypique")
            if assigned_c in clusters_diagnoses:
                final_clusters.append(assigned_c)
                assignment_sources.append("llm_reassigned")
            else:
                final_clusters.append("isole_atypique")
                assignment_sources.append("unclassified_outlier")

    assigned_df["cluster_id"] = final_clusters
    assigned_df["assignment_source"] = assignment_sources

    # Ajout du diagnostic pour le groupe 'isole_atypique' s'il contient des incidents
    if "isole_atypique" in set(final_clusters):
        clusters_diagnoses["isole_atypique"] = {
            "problem_title": "Incidents isolés et anomalies atypiques",
            "problem_description": "Ensemble d'incidents singuliers ne correspondant à aucun cluster récurrent établi.",
            "common_cause": "Origines variées et non récurrentes (facteurs externes exceptionnels, signaux faibles).",
            "system_impact": "Impacts ponctuels sans tendance systémique identifiée.",
            "actionable_recommendations": "Maintien d'une veille opérationnelle et traitement au cas par cas.",
        }

    # Fusion avec le dataframe original
    df_merged = df.merge(
        assigned_df[[cols["id"], "cluster_id", "assignment_source"]],
        on=cols["id"],
    )

    print(f"\n[5/6] Calcul de la criticité financière et opérationnelle (fréquence x coûts)...")
    crit_df = compute_cluster_criticality(
        df_merged,
        cluster_col="cluster_id",
        cost_col=cols["cost"],
        downtime_col=cols["downtime"],
    )

    # Intégration des diagnostics LLM dans la table récapitulative
    diag_rows = []
    for c_id in crit_df["cluster_id"]:
        d = clusters_diagnoses.get(c_id, {
            "problem_title": f"Groupe {c_id}",
            "problem_description": "N/A",
            "common_cause": "N/A",
            "system_impact": "N/A",
            "actionable_recommendations": "N/A",
        })
        diag_rows.append(d)

    diag_df = pd.DataFrame(diag_rows)
    clusters_summary = pd.concat([crit_df, diag_df], axis=1)

    # Réorganisation ordonnée des colonnes du résumé
    summary_columns_order = [
        "cluster_id",
        "criticality_rank",
        "criticality_tier",
        "problem_title",
        "common_cause",
        "frequency",
        "mean_cost_eur",
        "total_cost_eur",
        "mean_downtime_hours",
        "total_downtime_hours",
        "problem_description",
        "system_impact",
        "actionable_recommendations",
    ]
    summary_columns_order = [c for c in summary_columns_order if c in clusters_summary.columns]
    clusters_summary = clusters_summary[summary_columns_order]

    # Enrichissement du DataFrame d'incidents
    title_map = clusters_summary.set_index("cluster_id")["problem_title"].to_dict()
    cause_map = clusters_summary.set_index("cluster_id")["common_cause"].to_dict()
    tier_map = clusters_summary.set_index("cluster_id")["criticality_tier"].to_dict()

    df_merged["problem_title"] = df_merged["cluster_id"].map(title_map)
    df_merged["common_cause"] = df_merged["cluster_id"].map(cause_map)
    df_merged["cluster_criticality_tier"] = df_merged["cluster_id"].map(tier_map)

    # Réorganisation des colonnes enrichies
    leading_cols = [cols["id"], "cluster_id", "problem_title", "common_cause", "cluster_criticality_tier", "assignment_source"]
    remaining_cols = [c for c in df_merged.columns if c not in leading_cols]
    incidents_segmented = df_merged[leading_cols + remaining_cols]

    print(f"\n[6/6] Sauvegarde des fichiers dans {out_dir}...")
    summary_path = out_dir / "clusters_summary.csv"
    incidents_path = out_dir / "incidents_segmented.csv"

    clusters_summary.to_csv(summary_path, index=False)
    incidents_segmented.to_csv(incidents_path, index=False)

    print(f"       ✓ {summary_path} ({len(clusters_summary)} clusters)")
    print(f"       ✓ {incidents_path} ({len(incidents_segmented)} incidents)")

    # Affichage du top des clusters les plus critiques
    print("\n" + "=" * 90)
    print("TOP DES CLUSTERS PAR CRITICITÉ OPÉRATIONNELLE (FRÉQUENCE × COÛTS) :")
    print("=" * 90)
    top_crit = clusters_summary.head(8)
    for _, row in top_crit.iterrows():
        print(f"[{row['criticality_tier'].upper()}] Rang {row['criticality_rank']} | {row['cluster_id']} : {row['problem_title']}")
        print(f"    Impact : {row['frequency']} incidents | Coût total : {row['total_cost_eur']:,.2f} € (Moyenne : {row['mean_cost_eur']:,.2f} €) | Arrêt total : {row['total_downtime_hours']:.1f} h")
        print(f"    Cause commune : {row['common_cause']}")
        print(f"    Actions recommandées : {row['actionable_recommendations']}")
        print("-" * 90)

    return summary_path, incidents_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline agentique de segmentation des incidents")
    parser.add_argument("--input", type=Path, default=None, help="Chemin du CSV d'incidents")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Dossier de sortie")
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE, help="Taille min de cluster")
    parser.add_argument("--no-split", action="store_true", help="Clustering global sans découpage par asset_type")
    args = parser.parse_args()

    split_col = None if args.no_split else DEFAULT_SPLIT_COL
    run_agentic_segmentation(
        input_csv=args.input,
        out_dir=args.out_dir,
        min_cluster_size=args.min_cluster_size,
        split_col=split_col,
    )


if __name__ == "__main__":
    main()
