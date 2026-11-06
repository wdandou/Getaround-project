"""
Getaround - Web Dashboard (Streamlit)
======================================

Dashboard interactif destiné à l'équipe Product Management pour évaluer
l'impact d'un délai minimum entre deux locations, selon le seuil
(threshold) et le périmètre (scope) choisis.

Lancement :
    streamlit run app_streamlit.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "eda"))
from delay_analysis_eda import (  # noqa: E402
    build_chained_rentals,
    compute_kpis,
    load_data,
    simulate_thresholds,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "get_around_delay_analysis.xlsx"

st.set_page_config(page_title="Getaround - Delay Analysis", page_icon="🚗", layout="wide")


@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    chained = build_chained_rentals(df)
    kpis = compute_kpis(df)
    return df, chained, kpis


df, chained, kpis = get_data()

st.title("🚗 Getaround — Analyse des retards & simulation du délai minimum")
st.caption(
    "Aide à la décision pour le Product Management : faut-il instaurer un "
    "délai minimum entre deux locations, et pour quel périmètre de "
    "voitures ?"
)

# ---------------------------------------------------------------- KPIs ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Locations totales", f"{kpis['total_rentals']:,}")
col2.metric("Taux d'annulation", f"{kpis['cancellation_rate_pct']} %")
col3.metric("Locations en retard au checkout", f"{kpis['share_late_checkout_pct']} %")
col4.metric("Retard médian (locations en retard)", f"{kpis['median_delay_late_checkouts_min']:.0f} min")

st.divider()

# ------------------------------------------------------ Simulation UI ----
st.subheader("🎚️ Simulation du délai minimum entre deux locations")

c1, c2 = st.columns(2)
with c1:
    scope = st.radio(
        "Périmètre (scope)",
        options=["all", "connect"],
        format_func=lambda x: "Toutes les voitures" if x == "all" else "Voitures Connect uniquement",
        horizontal=True,
    )
with c2:
    threshold = st.slider("Seuil minimum (threshold, en minutes)", 0, 360, 60, step=15)

sim = simulate_thresholds(chained, total_rentals=len(df))
row = sim[(sim["scope"] == scope) & (sim["threshold_min"] == threshold)].iloc[0]

k1, k2, k3 = st.columns(3)
k1.metric("Locations bloquées", f"{int(row['rentals_blocked']):,}")
k2.metric("Part du total des locations (proxy revenu)", f"{row['share_of_all_rentals_%']} %")
k3.metric("Cas problématiques résolus", f"{row['pct_problematic_cases_solved']} %")

st.divider()

# --------------------------------------------------------------- Charts --
st.subheader("📊 Impact selon le seuil et le périmètre")

tab1, tab2, tab3 = st.tabs(["Locations impactées", "Frictions résolues", "Compromis"])

with tab1:
    fig = px.line(
        sim, x="threshold_min", y="share_of_all_rentals_%", color="scope", markers=True,
        labels={"threshold_min": "Seuil (min)", "share_of_all_rentals_%": "% de toutes les locations bloquées", "scope": "Périmètre"},
        title="Part des locations bloquées selon le seuil",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(
        sim, x="threshold_min", y="pct_problematic_cases_solved", color="scope", markers=True,
        labels={"threshold_min": "Seuil (min)", "pct_problematic_cases_solved": "% de cas problématiques résolus", "scope": "Périmètre"},
        title="Frictions résolues selon le seuil",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.scatter(
        sim, x="share_of_all_rentals_%", y="pct_problematic_cases_solved", color="scope",
        text="threshold_min",
        labels={"share_of_all_rentals_%": "% de locations bloquées (proxy revenu)", "pct_problematic_cases_solved": "% de frictions résolues", "scope": "Périmètre"},
        title="Compromis impact revenu / frictions résolues",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("📋 Table de simulation complète")
st.dataframe(sim, use_container_width=True, hide_index=True)

with st.expander("ℹ️ Méthodologie"):
    st.markdown(
        """
- **Locations bloquées** : locations "enchaînées" (même voiture, location
  immédiatement précédente identifiée) dont le délai planifié avec la
  location précédente (`time_delta_with_previous_rental_in_minutes`) est
  inférieur au seuil choisi. Elles n'auraient pas pu être réservées telles
  quelles si la fonctionnalité était active.
- **Cas problématiques** : parmi les locations enchaînées dont on connaît
  le retard réel du conducteur précédent, celles où le conducteur suivant
  a effectivement dû attendre (retard réel > délai planifié).
- **Cas résolus** : cas problématiques qui auraient été empêchés par le
  seuil choisi (le délai planifié était inférieur au seuil, donc la
  réservation n'aurait pas été proposée).
- **Part du revenu affecté** : faute d'un identifiant commun entre le jeu
  de données "retards" et le jeu de données "pricing", la part des
  locations bloquées est utilisée comme **proxy** de la part du revenu
  potentiellement affecté.
        """
    )
