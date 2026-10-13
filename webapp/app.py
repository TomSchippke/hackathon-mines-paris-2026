"""Application web Streamlit pour visualiser et explorer la segmentation des incidents."""

from __future__ import annotations

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
SUMMARY_PATH = Path("outputs/segmentation/clusters_summary.csv")
INCIDENTS_PATH = Path("outputs/segmentation/incidents_segmented.csv")

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
            "Exécutez d'abord : `python main.py --steps segmentation`"
        )
        st.stop()

    summary_df = pd.read_csv(SUMMARY_PATH)
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
        "- `clusters_summary.csv` (diagnostic LLM & Pareto)\n"
        "- `incidents_segmented.csv` (1800 incidents enrichis)"
    )

    # En-tête principal
    st.title("Tableau de Bord de Diagnostic des Incidents")
    st.markdown(
        "Visualisation de la segmentation non supervisée couplée au diagnostic causal "
        "d'agents IA et à l'évaluation de criticité opérationnelle."
    )

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs([
        "📊 Synthèse Exécutive & Pareto",
        "🔍 Diagnostic Détaillé par Cluster",
        "📋 Explorateur d'Incidents",
    ])

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
        st.markdown("#### Liste Complète des 37 Clusters de Défaillances")
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
            st.success(f"**Actions correctives / préventives préconisées :**\n\n{cluster_data['actionable_recommendations']}")

        st.divider()

        # Liste des incidents de ce cluster
        st.markdown(f"#### Incidents rattachés à ce cluster ({len(cluster_incidents)})")

        source_counts = cluster_incidents["assignment_source"].value_counts().to_dict()
        st.caption(
            f"Origine des affectations : "
            f"**{source_counts.get('dense_cluster', 0)}** par clustering dense HDBSCAN, "
            f"**{source_counts.get('llm_reassigned', 0)}** réassignés par l'agent LLM (Passe 2)."
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


if __name__ == "__main__":
    main()
