"""
Getaround - Analyse des retards au checkout (Delay Analysis)
=============================================================

Objectif : fournir les indicateurs nécessaires à la décision du Product
Manager sur l'instauration d'un délai minimum entre deux locations
(threshold) et sur son périmètre d'application (scope : toutes les
voitures ou uniquement les voitures "Connect").

Questions auxquelles ce script répond :
    1. Quelle part des locations (proxy du revenu) serait affectée par
       la fonctionnalité, selon le seuil et le périmètre choisis ?
    2. Combien de locations seraient affectées selon threshold / scope ?
    3. À quelle fréquence les conducteurs sont-ils en retard au
       checkout, et quel est l'impact sur le conducteur suivant ?
    4. Combien de cas problématiques (frictions) seraient résolus selon
       le seuil choisi ?

Le script :
    - charge et nettoie les données,
    - calcule les indicateurs clés (KPI),
    - simule l'impact de différents seuils / périmètres,
    - exporte les résultats (CSV) et les graphiques (PNG) utilisés dans
      le dashboard Streamlit et dans le document explicatif.

Données source :
https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "get_around_delay_analysis.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLDS_MIN = [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Getaround delay analysis Excel export."""
    return pd.read_excel(path)


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the high level KPIs used across the dashboard and reports."""
    ended = df[df["state"] == "ended"]
    known_delay = ended["delay_at_checkout_in_minutes"].notna()
    late = ended.loc[known_delay & (ended["delay_at_checkout_in_minutes"] > 0)]

    kpis = {
        "total_rentals": len(df),
        "ended_rentals": len(ended),
        "canceled_rentals": int((df["state"] == "canceled").sum()),
        "cancellation_rate_pct": round(100 * (df["state"] == "canceled").mean(), 1),
        "share_late_checkout_pct": round(100 * len(late) / known_delay.sum(), 1),
        "median_delay_late_checkouts_min": float(late["delay_at_checkout_in_minutes"].median()),
        "mean_delay_late_checkouts_min": float(late["delay_at_checkout_in_minutes"].mean()),
        "share_connect_pct": round(100 * (df["checkin_type"] == "connect").mean(), 1),
        "share_mobile_pct": round(100 * (df["checkin_type"] == "mobile").mean(), 1),
    }
    return kpis


def build_chained_rentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the subset of rentals that immediately follow another rental of
    the same car ("chained" rentals), and flag whether the next driver was
    actually impacted by a late checkout of the previous driver.
    """
    chained = df[df["previous_ended_rental_id"].notna()].copy()

    delay_map = df.set_index("rental_id")["delay_at_checkout_in_minutes"]
    chained["previous_delay_at_checkout"] = chained["previous_ended_rental_id"].map(delay_map)

    # Real gap the next driver actually had, once the previous driver's
    # own delay is taken into account (negative => next driver waited).
    chained["actual_gap_after_previous_checkout"] = (
        chained["time_delta_with_previous_rental_in_minutes"]
        - chained["previous_delay_at_checkout"].fillna(0)
    )
    chained["is_impacted"] = chained["actual_gap_after_previous_checkout"] < 0
    return chained


def simulate_thresholds(
    chained: pd.DataFrame, total_rentals: int, thresholds=THRESHOLDS_MIN
) -> pd.DataFrame:
    """
    For each (scope, threshold) combination, estimate:
      - how many / what share of rentals would be blocked (revenue proxy),
      - how many of the known friction cases ("is_impacted") would have
        been prevented.
    """
    known = chained[chained["previous_delay_at_checkout"].notna()]

    rows = []
    scopes = {
        "all": (chained, known),
        "connect": (
            chained[chained["checkin_type"] == "connect"],
            known[known["checkin_type"] == "connect"],
        ),
    }

    for scope_name, (scope_df, scope_known) in scopes.items():
        n_impacted = scope_known["is_impacted"].sum()
        for t in thresholds:
            blocked = scope_df[scope_df["time_delta_with_previous_rental_in_minutes"] < t]
            solved = scope_known[
                scope_known["is_impacted"]
                & (scope_known["time_delta_with_previous_rental_in_minutes"] < t)
            ]
            rows.append(
                {
                    "scope": scope_name,
                    "threshold_min": t,
                    "rentals_blocked": len(blocked),
                    "share_of_all_rentals_%": round(100 * len(blocked) / total_rentals, 2),
                    "share_of_chained_rentals_%": round(
                        100 * len(blocked) / len(scope_df), 2
                    )
                    if len(scope_df)
                    else 0,
                    "problematic_cases_solved": len(solved),
                    "pct_problematic_cases_solved": round(
                        100 * len(solved) / n_impacted, 1
                    )
                    if n_impacted
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def make_charts(df: pd.DataFrame, sim: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    # 1. Rentals by checkin type and state
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.crosstab(df["checkin_type"], df["state"]).plot(
        kind="bar", stacked=True, ax=ax, color=["#2E86AB", "#E76F51"]
    )
    ax.set_title("Locations par type de check-in et statut")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre de locations")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_dir / "01_state_by_checkin_type.png")
    plt.close(fig)

    # 2. Checkout delay distribution
    ended = df[(df["state"] == "ended") & df["delay_at_checkout_in_minutes"].notna()]
    clipped = ended["delay_at_checkout_in_minutes"].clip(-120, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clipped, bins=60, color="#2E86AB")
    ax.axvline(0, color="red", linestyle="--", label="À l'heure")
    ax.set_title("Distribution du retard au checkout (minutes)")
    ax.set_xlabel("Retard au checkout (minutes)")
    ax.set_ylabel("Nombre de locations")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "02_checkout_delay_distribution.png")
    plt.close(fig)

    # 3. Share of rentals blocked vs threshold
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for scope, color in [("all", "#2E86AB"), ("connect", "#E76F51")]:
        sub = sim[sim["scope"] == scope]
        ax.plot(
            sub["threshold_min"],
            sub["share_of_all_rentals_%"],
            marker="o",
            label=f"scope = {scope}",
            color=color,
        )
    ax.set_title("Part des locations impactées selon le seuil")
    ax.set_xlabel("Seuil minimum entre deux locations (minutes)")
    ax.set_ylabel("% de toutes les locations bloquées")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "03_share_rentals_blocked_vs_threshold.png")
    plt.close(fig)

    # 4. Share of problematic cases solved vs threshold
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for scope, color in [("all", "#2E86AB"), ("connect", "#E76F51")]:
        sub = sim[sim["scope"] == scope]
        ax.plot(
            sub["threshold_min"],
            sub["pct_problematic_cases_solved"],
            marker="o",
            label=f"scope = {scope}",
            color=color,
        )
    ax.set_title("Part des frictions résolues selon le seuil")
    ax.set_xlabel("Seuil minimum entre deux locations (minutes)")
    ax.set_ylabel("% de cas problématiques résolus")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "04_pct_problems_solved_vs_threshold.png")
    plt.close(fig)

    # 5. Trade-off scatter
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for scope, color in [("all", "#2E86AB"), ("connect", "#E76F51")]:
        sub = sim[sim["scope"] == scope]
        ax.plot(
            sub["share_of_all_rentals_%"],
            sub["pct_problematic_cases_solved"],
            marker="o",
            label=f"scope = {scope}",
            color=color,
        )
        for _, row in sub.iterrows():
            if row["threshold_min"] in (60, 120, 180, 240):
                ax.annotate(
                    f"{int(row['threshold_min'])}min",
                    (row["share_of_all_rentals_%"], row["pct_problematic_cases_solved"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
    ax.set_title("Compromis : impact revenu vs. frictions résolues")
    ax.set_xlabel("% de locations bloquées (proxy impact revenu)")
    ax.set_ylabel("% de cas problématiques résolus")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "05_tradeoff_scatter.png")
    plt.close(fig)


def main():
    df = load_data()
    kpis = compute_kpis(df)
    print("=== KPIs ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    chained = build_chained_rentals(df)
    sim = simulate_thresholds(chained, total_rentals=len(df))
    sim.to_csv(OUTPUT_DIR / "threshold_simulation_results.csv", index=False)
    print("\n=== Threshold / scope simulation (extract) ===")
    print(sim[sim["threshold_min"].isin([60, 120, 180])].to_string(index=False))

    make_charts(df, sim)
    print(f"\nGraphiques et résultats exportés dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
