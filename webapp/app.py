"""Application web Streamlit pour visualiser et explorer la segmentation des incidents."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Incident Intelligence | Segmentation & Criticité",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chemins des données
SEG_DIR = Path("outputs/segmentation")
SUMMARY_PATH = SEG_DIR / "clusters_merged.csv"
FINE_PATH = SEG_DIR / "clusters_fine.csv"
CONCENTRATION_PATH = SEG_DIR / "concentration.csv"
PRECURSOR_PAIRS_PATH = SEG_DIR / "precursor_pairs.csv"
PRECURSOR_EVIDENCE_PATH = SEG_DIR / "precursor_evidence.csv"
RUN_CMD = "python main.py --steps segmentation equipment"
NO_PRECURSOR_MSG = (
    "Aucun signal précurseur statistiquement significatif détecté sur la flotte — "
    "les incidents de ce type ne sont pas précédés plus souvent que par hasard "
    "d'un autre type d'incident sur le même équipement."
)
ISOLATED_ID = "isole_atypique"
SCOPE_BADGES = {"fleet": "🌐 flotte", "site": "📍 site", "equipment": "⚙️ équipement"}
INCIDENTS_PATH = Path("outputs/segmentation/incidents_segmented.csv")
EQUIPMENT_SUMMARY_PATH = Path("outputs/equipment/equipment_summary.csv")
EQUIPMENT_TIMELINE_PATH = Path("outputs/equipment/incidents_with_timeline.csv")
TECHNICIAN_PERF_PATH = Path("outputs/equipment/technician_performance.csv")

# Palette de couleurs par criticité
TIER_COLORS = {
    "Critique": "#EF4444",   # Rouge
    "Élevée": "#F97316",     # Orange
    "Moyenne": "#EAB308",    # Jaune
    "Faible": "#10B981",     # Vert
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données de segmentation avec mise en cache Streamlit."""
    if not SUMMARY_PATH.exists() or not INCIDENTS_PATH.exists():
        st.error(
            f"Fichiers introuvables ({SUMMARY_PATH} ou {INCIDENTS_PATH}). "
            f"Exécutez d'abord : `{RUN_CMD}`"
        )
        st.stop()

    summary_df = pd.read_csv(SUMMARY_PATH, keep_default_na=False, na_values=[""])
    summary_df["partition"] = summary_df["partition"].fillna("")
    # Alias de compatibilité : l'identifiant de cluster est l'id fusionné.
    summary_df["cluster_id"] = summary_df["merged_cluster_id"]
    incidents_df = pd.read_csv(INCIDENTS_PATH)

    # Conversion des types numériques
    for col in ["frequency", "criticality_rank"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    for col in ["mean_cost_eur", "total_cost_eur", "mean_downtime_hours", "total_downtime_hours"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0)

    for col in ["estimated_cost_eur", "downtime_hours"]:
        if col in incidents_df.columns:
            incidents_df[col] = pd.to_numeric(incidents_df[col], errors="coerce").fillna(0.0)

    return summary_df, incidents_df


def parse_json_list(value) -> list:
    """Parse une colonne JSON en liste ; [] si NaN, vide ou invalide."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


@st.cache_data
def load_action_plan_data() -> dict[str, pd.DataFrame]:
    """Charge fins, concentration, précurseurs et preuves (tolère les fichiers vides)."""
    out: dict[str, pd.DataFrame] = {}
    for key, path in {
        "fine": FINE_PATH,
        "concentration": CONCENTRATION_PATH,
        "pairs": PRECURSOR_PAIRS_PATH,
        "evidence": PRECURSOR_EVIDENCE_PATH,
    }.items():
        if not path.exists():
            st.error(f"Fichier introuvable : {path}. Exécutez d'abord : `{RUN_CMD}`")
            out[key] = pd.DataFrame()
            continue
        try:
            out[key] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            out[key] = pd.DataFrame()
    return out


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1"])


def render_action_card(row: pd.Series, data: dict[str, pd.DataFrame], incidents_df: pd.DataFrame) -> None:
    cid = row["merged_cluster_id"]
    st.markdown(f"**Partition :** `{row['partition']}` · **Identifiant :** `{cid}` · "
                f"**Arrêt cumulé :** {row['total_downtime_hours']:.0f} h")

    st.markdown("##### 🔧 Remédiation / cause")
    st.write(row.get("problem_description", ""))
    st.info(f"**Cause commune :** {row.get('common_cause', '')}")

    st.markdown("##### 🛡️ Défaillances évitées")
    outcomes = parse_json_list(row.get("prevented_outcomes"))
    if outcomes:
        st.markdown("\n".join(f"- {o}" for o in outcomes))
    else:
        st.caption("Non renseigné.")

    st.markdown("##### ✅ Recommandations")
    recos = [r for r in parse_json_list(row.get("recommendations")) if isinstance(r, dict)]
    if not recos:
        st.caption("Aucune recommandation.")
    for r in recos:
        badge = SCOPE_BADGES.get(r.get("scope"), str(r.get("scope", "")))
        targets = r.get("targets") if isinstance(r.get("targets"), list) else []
        tgt = f" — cibles : {', '.join(f'`{t}`' for t in targets)}" if targets else ""
        st.markdown(f"- **{badge}** {r.get('action', '')}{tgt}")
        if r.get("rationale"):
            st.caption(r["rationale"])

    st.markdown("##### 🎯 Où agir")
    conc = data["concentration"]
    if not conc.empty and "merged_cluster_id" in conc.columns:
        c = conc[(conc["merged_cluster_id"] == cid) & _bool_series(conc, "significant")]
    else:
        c = pd.DataFrame()
    if c.empty:
        st.caption("Aucune concentration significative (pas de site, équipement ou période sur-représenté).")
    else:
        st.dataframe(
            c.sort_values("lift", ascending=False)[["dimension", "value", "observed", "expected", "lift", "q_value"]]
            .rename(columns={"dimension": "Dimension", "value": "Valeur", "observed": "Observé",
                             "expected": "Attendu", "lift": "Lift", "q_value": "q"}),
            hide_index=True, use_container_width=True,
            column_config={"Attendu": st.column_config.NumberColumn(format="%.1f"),
                           "Lift": st.column_config.NumberColumn(format="%.2f"),
                           "q": st.column_config.NumberColumn(format="%.4f")},
        )

    st.markdown("##### ⏳ Signaux précurseurs")
    pairs = data["pairs"]
    rules = pd.DataFrame()
    if not pairs.empty:
        mask = _bool_series(pairs, "significant") & _bool_series(pairs, "plausible")
        into = (pairs.get("consequent") == cid) | (pairs.get("scope_cluster") == cid)
        rules = pairs[mask & into]
    if rules.empty:
        st.info(NO_PRECURSOR_MSG)
    else:
        for _, p in rules.sort_values("lift", ascending=False).iterrows():
            st.markdown(
                f"- **{p['kind']}** : `{p['antecedent']}` → `{p['consequent']}` sous {p['window_days']} j — "
                f"lift {p['lift']:.2f}, {p['n_equipments']} équipement(s), q={p['q_value']:.3f}"
            )
            if isinstance(p.get("monitoring_rule"), str):
                st.caption(f"Règle de surveillance : {p['monitoring_rule']}")
            if isinstance(p.get("mechanism"), str):
                st.caption(f"Mécanisme : {p['mechanism']}")
        ev = data["evidence"]
        if not ev.empty and "consequent_incident_id" in ev.columns:
            ids = set(incidents_df.loc[incidents_df["cluster_id"] == cid, "incident_id"])
            ev_c = ev[ev["consequent_incident_id"].isin(ids)]
            st.markdown(f"**Preuves : {len(ev_c)} cas observé(s)**")
            if not ev_c.empty:
                st.dataframe(ev_c, hide_index=True, use_container_width=True)
        else:
            st.caption("Aucune preuve détaillée disponible.")

    st.markdown("##### 🧩 Modes de défaillance")
    fine = data["fine"]
    if not fine.empty and "merged_cluster_id" in fine.columns:
        f = fine[fine["merged_cluster_id"] == cid]
    else:
        f = pd.DataFrame()
    if f.empty:
        st.caption("Aucun sous-cluster fin (incidents isolés ou non regroupés).")
    else:
        st.dataframe(
            f[["fine_cluster_id", "fine_title", "size", "keywords"]]
            .rename(columns={"fine_cluster_id": "ID", "fine_title": "Mode", "size": "Taille", "keywords": "Mots-clés"}),
            hide_index=True, use_container_width=True,
        )


@st.cache_data
def load_equipment_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les données chronologiques d'équipements ou les calcule automatiquement à la volée."""
    if not (EQUIPMENT_SUMMARY_PATH.exists() and EQUIPMENT_TIMELINE_PATH.exists() and TECHNICIAN_PERF_PATH.exists()):
        from equipment_analysis.pipeline import run_equipment_analysis
        res = run_equipment_analysis()
        return res["equipment_df"], res["timeline_df"], res["technician_df"]

    eq_df = pd.read_csv(EQUIPMENT_SUMMARY_PATH)
    tl_df = pd.read_csv(EQUIPMENT_TIMELINE_PATH)
    tech_df = pd.read_csv(TECHNICIAN_PERF_PATH)
    return eq_df, tl_df, tech_df


def format_currency(val: float) -> str:
    return f"{val:,.0f} €".replace(",", " ")


def main() -> None:
    summary_df, incidents_df = load_data()

    # Barre latérale
    st.sidebar.title("⚡ Incident Intelligence")
    st.sidebar.markdown(
        "**Pipeline Agentique de Diagnostic & Criticité**\n\n"
        "Analyse causale automatique et segmentation financière (fréquence × coûts)."
    )
    st.sidebar.divider()

    # Métriques rapides dans la barre latérale
    total_cost = summary_df["total_cost_eur"].sum()
    total_incidents = len(incidents_df)
    total_clusters = len(summary_df)
    crit_clusters = (summary_df["criticality_tier"] == "Critique").sum()

    st.sidebar.metric("Incidents analysés", f"{total_incidents:,}".replace(",", " "))
    st.sidebar.metric("Clusters identifiés", f"{total_clusters}")
    st.sidebar.metric("Impact financier total", format_currency(total_cost))
    st.sidebar.metric("Clusters Critiques", f"{crit_clusters} (Pareto)")

    st.sidebar.divider()
    st.sidebar.info(
        "💡 **Sources des données** :\n"
        "- `clusters_merged.csv` (plan d'action par cluster fusionné)\n"
        "- `incidents_segmented.csv` (1800 incidents enrichis)"
    )

    # En-tête principal
    st.title("Tableau de Bord de Diagnostic des Incidents")
    st.markdown(
        "Visualisation de la segmentation non supervisée couplée au diagnostic causal "
        "d'agents IA et à l'évaluation de criticité opérationnelle."
    )

    # Onglets principaux
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Plan d'action",
        "📊 Synthèse Exécutive & Pareto",
        "🔍 Diagnostic Détaillé par Cluster",
        "📋 Explorateur d'Incidents",
        "🔬 Chronologie & Efficacité Équipements",
    ])

    # =========================================================================
    # ONGLET 0 : PLAN D'ACTION
    # =========================================================================
    with tab0:
        st.subheader("Plan d'action par voie de remédiation")
        ap_data = load_action_plan_data()
        cards = summary_df[summary_df["merged_cluster_id"] != ISOLATED_ID].sort_values("criticality_rank")
        iso = summary_df[summary_df["merged_cluster_id"] == ISOLATED_ID]
        if len(iso):
            st.caption(f"{int(iso['frequency'].sum())} incidents isolés / atypiques non regroupés (sans carte).")

        a1, a2, a3 = st.columns([1, 1, 2])
        with a1:
            part_f = st.multiselect("Partition (type d'actif) :", sorted(p for p in cards["partition"].unique() if p), key="ap_part")
        with a2:
            tier_f = st.multiselect("Criticité :", ["Critique", "Élevée", "Moyenne", "Faible"], key="ap_tier")
        with a3:
            q_f = st.text_input("Recherche dans le titre :", "", key="ap_q")
        if part_f:
            cards = cards[cards["partition"].isin(part_f)]
        if tier_f:
            cards = cards[cards["criticality_tier"].isin(tier_f)]
        if q_f.strip():
            cards = cards[cards["problem_title"].str.contains(q_f.strip(), case=False, na=False)]

        st.caption(f"{len(cards)} carte(s) affichée(s).")
        for _, row in cards.iterrows():
            header = (f"[{row['criticality_tier']}] {row['problem_title']} — "
                      f"{int(row['frequency'])} incidents — {format_currency(row['total_cost_eur'])}")
            with st.expander(header):
                render_action_card(row, ap_data, incidents_df)

    # =========================================================================
    # ONGLET 1 : SYNTHÈSE EXÉCUTIVE & PARETO
    # =========================================================================
    with tab1:
        st.subheader("Vue d'ensemble de la flotte et Analyse Pareto")

        # Cartes KPI
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Incidents", f"{total_incidents:,}".replace(",", " "))
        col2.metric("Impact Financier", format_currency(total_cost))
        col3.metric("Arrêt Cumulé", f"{incidents_df['downtime_hours'].sum():,.0f} h".replace(",", " "))
        col4.metric("Coût Moyen / Incident", format_currency(incidents_df['estimated_cost_eur'].mean()))
        col5.metric("Top 20% des Causes", f"{int(total_clusters * 0.2)} clusters")

        st.divider()

        # Graphique 1 : Courbe de Pareto (Coûts cumulés)
        sorted_clusters = summary_df.sort_values(by="total_cost_eur", ascending=False).copy()
        sorted_clusters["cum_cost"] = sorted_clusters["total_cost_eur"].cumsum()
        sorted_clusters["cum_ratio"] = (sorted_clusters["cum_cost"] / total_cost) * 100

        fig_pareto = go.Figure()
        # Barres de coûts
        fig_pareto.add_trace(go.Bar(
            x=sorted_clusters["cluster_id"],
            y=sorted_clusters["total_cost_eur"],
            name="Coût total (€)",
            marker=dict(
                color=[TIER_COLORS.get(t, "#64748B") for t in sorted_clusters["criticality_tier"]],
            ),
            hovertemplate="<b>%{x}</b><br>Coût: %{y:,.0f} €<extra></extra>",
            yaxis="y1",
        ))
        # Ligne de ratio cumulé
        fig_pareto.add_trace(go.Scatter(
            x=sorted_clusters["cluster_id"],
            y=sorted_clusters["cum_ratio"],
            name="% Coût cumulé",
            mode="lines+markers",
            line=dict(color="#2563EB", width=3),
            hovertemplate="<b>%{x}</b><br>Cumul: %{y:.1f}%<extra></extra>",
            yaxis="y2",
        ))
        # Seuil 80% Pareto
        fig_pareto.add_shape(
            type="line",
            x0=-0.5,
            x1=len(sorted_clusters) - 0.5,
            y0=80,
            y1=80,
            yref="y2",
            line=dict(color="#DC2626", dash="dash", width=2),
        )
        fig_pareto.update_layout(
            title="Analyse Pareto des Clusters (Règle 80/20)",
            xaxis=dict(title="Cluster ID", tickangle=-45),
            yaxis=dict(title="Coût total d'exploitation (€)"),
            yaxis2=dict(
                title="% Cumulé du coût total",
                overlaying="y",
                side="right",
                range=[0, 105],
                ticksuffix="%",
            ),
            legend=dict(x=0.8, y=1.15, orientation="h"),
            hovermode="x unified",
            height=460,
            margin=dict(l=40, r=40, t=60, b=80),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.divider()

        # Graphique 2 & 3 : Matrice 2D Risque & Répartition par Actif
        col_m1, col_m2 = st.columns([6, 5])

        with col_m1:
            st.markdown("#### Matrice Risque : Fréquence vs Coût Moyen")
            fig_matrix = px.scatter(
                summary_df,
                x="frequency",
                y="mean_cost_eur",
                size="total_downtime_hours",
                color="criticality_tier",
                color_discrete_map=TIER_COLORS,
                hover_name="problem_title",
                hover_data={
                    "cluster_id": True,
                    "frequency": True,
                    "mean_cost_eur": ":,.0f",
                    "total_cost_eur": ":,.0f",
                    "total_downtime_hours": ":,.1f",
                    "criticality_tier": False,
                },
                labels={
                    "frequency": "Fréquence (nombre d'incidents)",
                    "mean_cost_eur": "Coût moyen unitaire (€)",
                    "total_downtime_hours": "Arrêt total (heures)",
                    "criticality_tier": "Niveau de criticité",
                },
                size_max=35,
            )
            fig_matrix.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_matrix, use_container_width=True)

        with col_m2:
            st.markdown("#### Répartition Financière par Site et Domaine")
            site_cost = incidents_df.groupby(["site", "asset_type"])["estimated_cost_eur"].sum().reset_index()
            site_cost = site_cost.sort_values(by="estimated_cost_eur", ascending=True)

            fig_site = px.bar(
                site_cost,
                y="site",
                x="estimated_cost_eur",
                color="asset_type",
                orientation="h",
                color_discrete_map={"wind": "#0EA5E9", "solar": "#F59E0B"},
                labels={
                    "estimated_cost_eur": "Coût total d'exploitation (€)",
                    "site": "Site d'exploitation",
                    "asset_type": "Actif",
                },
            )
            fig_site.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_site, use_container_width=True)

        st.divider()

        # Tableau des 37 clusters
        st.markdown("#### Liste Complète des Clusters de Défaillances")
        tier_filter = st.multiselect(
            "Filtrer par niveau de criticité :",
            options=["Critique", "Élevée", "Moyenne", "Faible"],
            default=["Critique", "Élevée"],
        )
        filtered_summary = summary_df[summary_df["criticality_tier"].isin(tier_filter)] if tier_filter else summary_df

        display_cols = [
            "criticality_rank",
            "criticality_tier",
            "cluster_id",
            "problem_title",
            "frequency",
            "total_cost_eur",
            "mean_cost_eur",
            "total_downtime_hours",
            "common_cause",
        ]
        st.dataframe(
            filtered_summary[display_cols].style.format({
                "total_cost_eur": lambda x: f"{x:,.0f} €",
                "mean_cost_eur": lambda x: f"{x:,.0f} €",
                "total_downtime_hours": lambda x: f"{x:.1f} h",
            }),
            use_container_width=True,
            height=350,
        )

    # =========================================================================
    # ONGLET 2 : DIAGNOSTIC DÉTAILLÉ PAR CLUSTER
    # =========================================================================
    with tab2:
        st.subheader("Fiche Diagnostic & Cause Racine par Cluster")

        cluster_options = summary_df["cluster_id"].tolist()
        cluster_labels = {
            c: f"[Rang {summary_df.loc[summary_df['cluster_id'] == c, 'criticality_rank'].values[0]}] {c} - {summary_df.loc[summary_df['cluster_id'] == c, 'problem_title'].values[0]}"
            for c in cluster_options
        }

        selected_cluster = st.selectbox(
            "Sélectionnez un cluster à examiner :",
            options=cluster_options,
            format_func=lambda x: cluster_labels.get(x, x),
        )

        cluster_data = summary_df[summary_df["cluster_id"] == selected_cluster].iloc[0]
        cluster_incidents = incidents_df[incidents_df["cluster_id"] == selected_cluster]

        # En-tête du cluster sélectionné
        badge_color = TIER_COLORS.get(cluster_data["criticality_tier"], "#64748B")
        st.markdown(
            f"""
            <div style="padding: 18px; border-radius: 8px; border-left: 6px solid {badge_color}; background-color: #F8FAFC; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #0F172A;">{cluster_data['problem_title']}</h3>
                <span style="background-color: {badge_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem;">
                    NIVEAU : {cluster_data['criticality_tier'].upper()} (Rang {cluster_data['criticality_rank']})
                </span>
                <span style="margin-left: 10px; font-size: 0.9rem; color: #475569;">
                    Identifiant technique : <code>{cluster_data['cluster_id']}</code>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chiffres clés du cluster
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nombre d'incidents", f"{cluster_data['frequency']}")
        c2.metric("Coût d'exploitation total", format_currency(cluster_data['total_cost_eur']))
        c3.metric("Coût unitaire moyen", format_currency(cluster_data['mean_cost_eur']))
        c4.metric("Arrêt total cumulé", f"{cluster_data['total_downtime_hours']:.1f} h")

        st.markdown("---")

        # Diagnostic causal IA
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("#### 🔬 Analyse Causale (Agent IA)")
            st.info(f"**Cause racine commune identifiée :**\n\n{cluster_data['common_cause']}")
            st.markdown(f"**Description du phénomène :**\n\n{cluster_data['problem_description']}")

        with d_col2:
            st.markdown("#### 🛡️ Impact Système & Recommandations")
            st.warning(f"**Impact opérationnel sur l'installation :**\n\n{cluster_data['system_impact']}")
            recos = [r for r in parse_json_list(cluster_data["recommendations"]) if isinstance(r, dict)]
            reco_md = "\n".join(
                f"- **{SCOPE_BADGES.get(r.get('scope'), r.get('scope', ''))}** {r.get('action', '')}"
                + (f" ({', '.join(map(str, r['targets']))})" if isinstance(r.get("targets"), list) and r["targets"] else "")
                for r in recos
            ) or "Aucune recommandation."
            st.success(f"**Actions correctives / préventives préconisées :**\n\n{reco_md}")

        st.divider()

        # Liste des incidents de ce cluster
        st.markdown(f"#### Incidents rattachés à ce cluster ({len(cluster_incidents)})")

        source_counts = cluster_incidents["assignment_source"].value_counts().to_dict()
        st.caption(
            "Origine des affectations : "
            + (" · ".join(f"**{n}** {src}" for src, n in source_counts.items()) or "n/a")
        )

        show_incidents_cols = [
            "incident_id",
            "site",
            "equipment_id",
            "date",
            "severity",
            "estimated_cost_eur",
            "downtime_hours",
            "weather_condition",
            "assignment_source",
        ]
        show_incidents_cols = [c for c in show_incidents_cols if c in cluster_incidents.columns]

        st.dataframe(
            cluster_incidents[show_incidents_cols].style.format({
                "estimated_cost_eur": lambda x: f"{x:,.0f} €",
                "downtime_hours": lambda x: f"{x:.1f} h",
            }),
            use_container_width=True,
            height=250,
        )

        # Inspection approfondie d'un incident
        st.markdown("##### 🔍 Rapport d'intervention individuel")
        inc_ids = cluster_incidents["incident_id"].tolist()
        if inc_ids:
            selected_inc_id = st.selectbox(
                "Choisir un incident à lire en détail :",
                options=inc_ids,
                format_func=lambda x: f"Incident #{x} ({cluster_incidents.loc[cluster_incidents['incident_id'] == x, 'site'].values[0]})",
            )
            selected_inc = cluster_incidents[cluster_incidents["incident_id"] == selected_inc_id].iloc[0]
            with st.expander("Consulter le texte complet du rapport d'intervention", expanded=True):
                st.write(selected_inc["text"])
                st.caption(
                    f"Date : {selected_inc.get('date', 'N/A')} | "
                    f"Technicien : {selected_inc.get('technician_id', 'N/A')} | "
                    f"Format source : {selected_inc.get('document_format', 'N/A')}"
                )

    # =========================================================================
    # ONGLET 3 : EXPLORATEUR D'INCIDENTS
    # =========================================================================
    with tab3:
        st.subheader("Explorateur & Recherche Avancée d'Incidents")

        # Filtres horizontaux
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            search_query = st.text_input("🔎 Recherche plein texte dans les rapports :", "")
        with f_col2:
            asset_filter = st.multiselect(
                "Type d'actif :",
                options=sorted(incidents_df["asset_type"].dropna().unique()),
                default=sorted(incidents_df["asset_type"].dropna().unique()),
            )
        with f_col3:
            site_filter = st.multiselect(
                "Site :",
                options=sorted(incidents_df["site"].dropna().unique()),
            )
        with f_col4:
            crit_filter = st.multiselect(
                "Criticité du cluster :",
                options=["Critique", "Élevée", "Moyenne", "Faible"],
                default=["Critique", "Élevée", "Moyenne", "Faible"],
            )

        # Application des filtres
        filtered_df = incidents_df.copy()
        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df["text"].str.contains(search_query.strip(), case=False, na=False)
            ]
        if asset_filter:
            filtered_df = filtered_df[filtered_df["asset_type"].isin(asset_filter)]
        if site_filter:
            filtered_df = filtered_df[filtered_df["site"].isin(site_filter)]
        if crit_filter:
            filtered_df = filtered_df[filtered_df["cluster_criticality_tier"].isin(crit_filter)]

        # Métriques sur la sélection
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Résultats filtrés", f"{len(filtered_df)} / {len(incidents_df)}")
        m2.metric("Impact financier filtré", format_currency(filtered_df["estimated_cost_eur"].sum()))
        m3.metric("Arrêt cumulé filtré", f"{filtered_df['downtime_hours'].sum():,.0f} h".replace(",", " "))
        m4.metric("Clusters touchés", f"{filtered_df['cluster_id'].nunique()}")

        # Bouton d'export CSV
        csv_export = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger la sélection filtrée (CSV)",
            data=csv_export,
            file_name="incidents_selection_filtree.csv",
            mime="text/csv",
        )

        st.dataframe(
            filtered_df[[
                "incident_id",
                "cluster_id",
                "problem_title",
                "cluster_criticality_tier",
                "site",
                "equipment_id",
                "date",
                "severity",
                "estimated_cost_eur",
                "downtime_hours",
                "weather_condition",
                "assignment_source",
            ]].style.format({
                "estimated_cost_eur": lambda x: f"{x:,.0f} €",
                "downtime_hours": lambda x: f"{x:.1f} h",
            }),
            use_container_width=True,
            height=400,
        )

    # =========================================================================
    # ONGLET 4 : CHRONOLOGIE ÉQUIPEMENTS & EFFICACITÉ TECHNICIENS
    # =========================================================================
    with tab4:
        st.subheader("🔬 Chronologie des Pannes & Évaluation des Interventions")
        st.markdown(
            "Recentrage de l'analyse sur le cycle de vie des **actifs (`equipment_id`)** et "
            "l'efficacité opérationnelle des **techniciens**. "
            "Reconstitution des frises chronologiques, détection des récidives précoces et palmarès des *bad actors*."
        )

        eq_df, tl_df, tech_df = load_equipment_data()

        # Barre de filtres et seuil de récidive
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            eq_site_filter = st.selectbox(
                "Filtrer par site :",
                options=["Tous"] + sorted(eq_df["site"].dropna().unique().tolist()),
            )
        with fc2:
            eq_asset_filter = st.selectbox(
                "Filtrer par type d'actif :",
                options=["Tous"] + sorted(eq_df["asset_type"].dropna().unique().tolist()),
            )
        with fc3:
            rec_threshold = st.slider(
                "⏱️ Seuil temporel de récidive (jours après intervention) :",
                min_value=3,
                max_value=60,
                value=30,
                step=1,
                help="Un incident survenant moins de N jours après le précédent sur la même machine est classé comme récidive post-intervention.",
            )

        # Filtrage des DataFrames
        filtered_eq = eq_df.copy()
        if eq_site_filter != "Tous":
            filtered_eq = filtered_eq[filtered_eq["site"] == eq_site_filter]
        if eq_asset_filter != "Tous":
            filtered_eq = filtered_eq[filtered_eq["asset_type"] == eq_asset_filter]

        # Calcul dynamique du flag de récidive selon le curseur
        tl_work = tl_df.copy()
        tl_work["datetime"] = pd.to_datetime(tl_work["date"])
        tl_work["is_custom_rec"] = (
            tl_work["days_since_last_incident"].notna()
            & (tl_work["days_since_last_incident"] <= rec_threshold)
        )
        tl_work["followed_by_custom_rec"] = (
            tl_work["days_until_next_incident"].notna()
            & (tl_work["days_until_next_incident"] <= rec_threshold)
        )

        # KPIs globaux flotte
        st.divider()
        g1, g2, g3, g4 = st.columns(4)
        eq_ids_filtered = set(filtered_eq["equipment_id"])
        fleet_tl = tl_work[tl_work["equipment_id"].isin(eq_ids_filtered)]
        fleet_total_inc = len(fleet_tl)
        fleet_rec_cnt = int(fleet_tl["is_custom_rec"].sum())
        fleet_rec_rate = (fleet_rec_cnt / max(1, fleet_total_inc)) * 100
        fleet_mtbf = filtered_eq["mtbf_days"].mean() if not filtered_eq.empty else 0

        g1.metric("Équipements filtrés", f"{len(filtered_eq)} machines")
        g2.metric("MTBF Moyen Flotte", f"{fleet_mtbf:.1f} jours")
        g3.metric(f"Récidives (< {rec_threshold}j)", f"{fleet_rec_cnt} incidents")
        g4.metric(f"Taux de récidive global", f"{fleet_rec_rate:.1f} %")

        # Section 1 : Chronologie détaillée par équipement
        st.subheader("📅 Chronique de vie & Timeline par Équipement")

        selected_eq = st.selectbox(
            "Sélectionner un équipement pour visualiser sa timeline :",
            options=filtered_eq["equipment_id"].tolist(),
            format_func=lambda x: f"{x} ({filtered_eq.loc[filtered_eq['equipment_id'] == x, 'site'].values[0]} - {filtered_eq.loc[filtered_eq['equipment_id'] == x, 'total_incidents'].values[0]} pannes)",
        )

        if selected_eq:
            eq_info = filtered_eq[filtered_eq["equipment_id"] == selected_eq].iloc[0]
            eq_timeline = tl_work[tl_work["equipment_id"] == selected_eq].sort_values("datetime").copy()

            # Cartes métriques pour cet équipement
            c1, c2, c3, c4, c5 = st.columns(5)
            eq_rec_count = int(eq_timeline["is_custom_rec"].sum())
            eq_rec_rate = (eq_rec_count / max(1, len(eq_timeline) - 1)) * 100
            c1.metric("Total Pannes", f"{eq_info['total_incidents']}")
            c2.metric("MTBF (délai moyen)", f"{eq_info['mtbf_days']:.1f} j")
            c3.metric(f"Récidives (< {rec_threshold}j)", f"{eq_rec_count} ({eq_rec_rate:.0f} %)")
            c4.metric("Coût Cumulé", format_currency(eq_info["total_cost_eur"]))
            c5.metric("Technicien Fréquent", f"{eq_info['most_frequent_technician']}")

            # Graphique interactif de la frise chronologique
            eq_timeline["statut_panne"] = eq_timeline["is_custom_rec"].map(
                {True: f"⚠️ Récidive (< {rec_threshold}j)", False: "Panne isolée / 1ère panne"}
            )
            eq_timeline["date_str"] = eq_timeline["datetime"].dt.strftime("%Y-%m-%d")

            hover_cols = {
                "datetime": False,
                "date_str": True,
                "incident_order": True,
                "severity": True,
                "technician_id": True,
                "prev_technician_id": True,
                "days_since_last_incident": ":.1f",
                "downtime_hours": ":.1f",
                "estimated_cost_eur": ":.0f",
            }
            if "problem_title" in eq_timeline.columns:
                hover_cols["problem_title"] = True

            fig_timeline = px.scatter(
                eq_timeline,
                x="datetime",
                y="days_since_last_incident",
                color="statut_panne",
                size="downtime_hours",
                color_discrete_map={
                    f"⚠️ Récidive (< {rec_threshold}j)": "#EF4444",
                    "Panne isolée / 1ère panne": "#3B82F6",
                },
                hover_data=hover_cols,
                labels={
                    "datetime": "Date de l'incident",
                    "days_since_last_incident": "Jours depuis la panne précédente",
                    "statut_panne": "Statut de récidive",
                    "downtime_hours": "Arrêt (h)",
                    "date_str": "Date",
                    "technician_id": "Technicien",
                    "prev_technician_id": "Tech précédent",
                },
                title=f"Frise Chronologique des Pannes : {selected_eq} (Taille de bulle = Durée d'arrêt)",
            )
            fig_timeline.add_hline(
                y=rec_threshold,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Seuil de récidive ({rec_threshold}j)",
                annotation_position="bottom right",
            )
            fig_timeline.update_layout(height=420)
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Historique tabulaire
            with st.expander(f"📋 Détail de l'historique des {len(eq_timeline)} pannes de {selected_eq}"):
                display_cols = [
                    "incident_order",
                    "date",
                    "severity",
                    "days_since_last_incident",
                    "is_custom_rec",
                    "technician_id",
                    "prev_technician_id",
                    "downtime_hours",
                    "estimated_cost_eur",
                    "problem_title",
                ]
                avail_cols = [c for c in display_cols if c in eq_timeline.columns]
                st.dataframe(
                    eq_timeline[avail_cols].style.format({
                        "days_since_last_incident": lambda x: f"{x:.1f} j" if pd.notna(x) else "-",
                        "downtime_hours": lambda x: f"{x:.1f} h",
                        "estimated_cost_eur": lambda x: f"{x:,.0f} €",
                    }),
                    use_container_width=True,
                    height=250,
                )

        st.divider()

        # Section 2 : Palmarès des Bad Actors
        st.subheader("⚠️ Palmarès des « Bad Actors » (Équipements les plus fragiles)")
        st.markdown(
            "Les équipements combinant **faible MTBF**, **taux élevé de récidive** et **coûts cumulés élevés** "
            "doivent faire l'objet d'une révision majeure ou d'un remplacement préventif."
        )

        col_ba1, col_ba2 = st.columns([3, 2])
        with col_ba1:
            fig_scatter = px.scatter(
                filtered_eq,
                x="mtbf_days",
                y="total_cost_eur",
                size="total_incidents",
                color="bad_actor_score",
                color_continuous_scale="Reds",
                hover_name="equipment_id",
                hover_data=["site", "total_incidents", "recurrence_rate_30d", "bad_actor_score"],
                labels={
                    "mtbf_days": "MTBF moyen (Jours de bon fonctionnement)",
                    "total_cost_eur": "Coût cumulé total (€)",
                    "bad_actor_score": "Score Bad Actor (0-100)",
                    "total_incidents": "Nombre d'incidents",
                },
                title="Vulnérabilité : MTBF vs Coût Cumulé (Coloré par Score de Criticité)",
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_ba2:
            st.markdown("**Top 10 Actifs Critiques à Auditer :**")
            top_bad = filtered_eq.head(10)[[
                "equipment_id",
                "site",
                "total_incidents",
                *(["severe_incidents"] if "severe_incidents" in filtered_eq.columns else []),
                "mtbf_days",
                "recurrence_rate_30d",
                "total_cost_eur",
                "bad_actor_score",
            ]]
            st.dataframe(
                top_bad.style.format({
                    "mtbf_days": "{:.1f} j",
                    "recurrence_rate_30d": "{:.1f} %",
                    "total_cost_eur": lambda x: f"{x:,.0f} €",
                    "bad_actor_score": "{:.1f}",
                }),
                use_container_width=True,
                height=350,
            )

        st.divider()

        # Section 3 : Audit & Efficacité des Techniciens
        st.subheader("👷 Audit d'Efficacité des Interventions par Technicien")
        st.markdown(
            "Mesure la pérennité des réparations : quel est le **taux de récidive** post-intervention "
            "et le **temps moyen de survie de la machine** avant sa prochaine panne ?"
        )

        # Calcul dynamique des métriques techniciens avec le seuil personnalisé
        custom_tech_stats = []
        for tech_id, t_group in tl_work.groupby("technician_id"):
            tot = len(t_group)
            opps = t_group["days_until_next_incident"].dropna()
            recs = t_group["followed_by_custom_rec"].sum()
            surv = opps.mean() if len(opps) > 0 else float("nan")
            rate = (recs / len(opps) * 100) if len(opps) > 0 else 0.0
            custom_tech_stats.append({
                "technician_id": tech_id,
                "total_interventions": tot,
                f"recidives_{rec_threshold}j": int(recs),
                f"taux_recidive_{rec_threshold}j": round(rate, 1),
                "delai_survie_moyen_jours": round(surv, 1) if pd.notna(surv) else "-",
                "duree_arret_moyenne": round(t_group["downtime_hours"].mean(), 1),
                "cout_moyen": round(t_group["estimated_cost_eur"].mean(), 0),
            })
        custom_tech_df = pd.DataFrame(custom_tech_stats).sort_values(
            by=f"taux_recidive_{rec_threshold}j", ascending=False
        )

        t_col1, t_col2 = st.columns([3, 2])
        with t_col1:
            fig_tech = px.bar(
                custom_tech_df,
                x=f"taux_recidive_{rec_threshold}j",
                y="technician_id",
                orientation="h",
                color=f"taux_recidive_{rec_threshold}j",
                color_continuous_scale="RdYlGn_r",
                labels={
                    f"taux_recidive_{rec_threshold}j": f"Taux de panne sous {rec_threshold}j (%)",
                    "technician_id": "Technicien",
                },
                title=f"Taux de Récidive Post-Intervention sous {rec_threshold} jours par Technicien",
            )
            fig_tech.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
            st.plotly_chart(fig_tech, use_container_width=True)

        with t_col2:
            st.markdown(f"**Classement Qualité & Rappels (< {rec_threshold}j) :**")
            st.dataframe(
                custom_tech_df.style.format({
                    f"taux_recidive_{rec_threshold}j": "{:.1f} %",
                    "duree_arret_moyenne": "{:.1f} h",
                    "cout_moyen": lambda x: f"{x:,.0f} €",
                }),
                use_container_width=True,
                height=380,
            )


if __name__ == "__main__":
    main()

