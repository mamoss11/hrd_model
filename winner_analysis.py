# ─────────────────────────────────────────────────────────────
#  HRD Model — Historical Winner Analysis
#
#  Fetches Statcast attributes for past HRD winners in their
#  winning seasons, identifies common trends, and scores the
#  current field by similarity to the winner profile.
#
#  Usage:
#    python winner_analysis.py
#    python winner_analysis.py --suggest-weights
# ─────────────────────────────────────────────────────────────
import io
import sys
import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_fetcher import fetch_statcast, compute_attributes
from config import PLAYERS

# ── Past HRD Winners ─────────────────────────────────────────
# Add 2022 and 2023 once confirmed.
# season: year they won (used to pull correct Statcast window)
# bats:   handedness for pulled-barrel calculation
WINNERS = [
    {"name": "Aaron Judge",        "mlb_id": 592450, "season": 2017, "bats": "R"},
    {"name": "Bryce Harper",       "mlb_id": 547180, "season": 2018, "bats": "L"},
    {"name": "Pete Alonso",        "mlb_id": 624413, "season": 2019, "bats": "R"},
    {"name": "Pete Alonso",        "mlb_id": 624413, "season": 2021, "bats": "R"},
    {"name": "Juan Soto",          "mlb_id": 665742, "season": 2022, "bats": "L"},
    {"name": "Vlad Guerrero Jr.",  "mlb_id": 665489, "season": 2023, "bats": "R"},
    {"name": "Teoscar Hernandez",  "mlb_id": 606192, "season": 2024, "bats": "R"},
    {"name": "Cal Raleigh",        "mlb_id": 663728, "season": 2025, "bats": "L"},
]

# Attributes available pre-2024 (no bat speed in Statcast before 2024)
ATTRS_4 = ["max_exit_velo", "pct90_exit_velo", "pulled_barrel_pct", "hr_per_pa"]
ATTRS_5 = ["bat_speed", "max_exit_velo", "pct90_exit_velo", "pulled_barrel_pct", "hr_per_pa"]

SEASON_WINDOWS = {
    2017: ("2017-04-02", "2017-07-10"),
    2018: ("2018-03-29", "2018-07-10"),
    2019: ("2019-03-28", "2019-07-08"),
    2021: ("2021-04-01", "2021-07-12"),
    2022: ("2022-04-07", "2022-07-11"),
    2023: ("2023-03-30", "2023-07-10"),
    2024: ("2024-03-28", "2024-07-13"),
    2025: ("2025-03-27", "2025-07-14"),
}


def load_winner_attributes() -> list:
    enriched = []
    for w in WINNERS:
        start, end = SEASON_WINDOWS[w["season"]]
        print(f"  Fetching {w['name']} ({w['season']}) ...")
        try:
            df    = fetch_statcast(w["mlb_id"], start, end)
            attrs = compute_attributes(df, w["bats"])
        except Exception as exc:
            print(f"    WARNING: {exc}")
            attrs = {k: np.nan for k in ATTRS_5}
            attrs["total_balls_hit"] = 0
        enriched.append({**w, **attrs})
    return enriched


def load_current_field_attributes() -> list:
    from data_fetcher import load_player_attributes
    from config import SEASON_START, SEASON_END
    return load_player_attributes(PLAYERS, SEASON_START, SEASON_END)


def print_attribute_table(players: list, attrs: list, title: str) -> None:
    print(f"\n{title}")
    print("-" * 80)
    header = f"  {'Name':<24}" + "".join(f"{a:>14}" for a in attrs)
    print(header)
    print("-" * 80)
    for p in players:
        row = f"  {p['name']:<24}"
        for a in attrs:
            val = p.get(a, np.nan)
            if np.isnan(val):
                row += f"{'N/A':>14}"
            elif a in ("pulled_barrel_pct", "hr_per_pa"):
                row += f"{val:>13.3f} "
            else:
                row += f"{val:>13.1f} "
        print(row)


def winner_centroid(winners: list, attrs: list) -> dict:
    centroid = {}
    for a in attrs:
        vals = [w[a] for w in winners if not np.isnan(w.get(a, np.nan))]
        centroid[a] = float(np.mean(vals)) if vals else np.nan
    return centroid


def similarity_scores(field: list, centroid: dict, attrs: list) -> list:
    """
    Score each current player by similarity to winner centroid.
    Uses normalised Euclidean distance, converted to a 0-100 similarity score.
    """
    # Build a combined dataset for normalisation
    all_players = field
    col_min = {a: min(p.get(a, np.nan) for p in all_players if not np.isnan(p.get(a, np.nan))) for a in attrs}
    col_max = {a: max(p.get(a, np.nan) for p in all_players if not np.isnan(p.get(a, np.nan))) for a in attrs}

    def normalise(val, a):
        lo, hi = col_min[a], col_max[a]
        if hi == lo or np.isnan(val):
            return 0.5
        return (val - lo) / (hi - lo)

    # Normalise centroid
    c_norm = {a: normalise(centroid[a], a) for a in attrs}

    scored = []
    for p in field:
        p_norm = {a: normalise(p.get(a, np.nan), a) for a in attrs}
        dist   = np.sqrt(sum((p_norm[a] - c_norm[a]) ** 2 for a in attrs))
        score  = round(100 / (1 + dist), 1)
        scored.append({**p, "similarity": score, "_dist": dist})

    return sorted(scored, key=lambda x: -x["similarity"])


def suggest_weights(winners: list, attrs: list) -> dict:
    """
    Rank attributes by how much they differentiate winners from a
    random baseline (measured by z-score of winner values relative
    to the current field mean/std).
    """
    from config import PLAYERS
    from data_fetcher import load_player_attributes
    from config import SEASON_START, SEASON_END

    field = load_player_attributes(PLAYERS, SEASON_START, SEASON_END)
    field_means = {a: np.nanmean([p.get(a, np.nan) for p in field]) for a in attrs}
    field_stds  = {a: np.nanstd([p.get(a, np.nan) for p in field]) for a in attrs}

    z_scores = {}
    for a in attrs:
        vals = [w[a] for w in winners if not np.isnan(w.get(a, np.nan))]
        if not vals or field_stds[a] == 0:
            z_scores[a] = 0.0
            continue
        winner_mean = np.mean(vals)
        z_scores[a] = abs((winner_mean - field_means[a]) / field_stds[a])

    total = sum(z_scores.values())
    weights = {a: round(z_scores[a] / total, 3) if total > 0 else 0.2 for a in attrs}
    return weights, z_scores, field_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggest-weights", action="store_true",
                        help="Suggest attribute weights based on winner z-scores")
    args = parser.parse_args()

    print("\nFetching historical winner data...")
    winners = load_winner_attributes()

    # Separate into 4-attr (all winners) and 5-attr (2024+ only)
    winners_5 = [w for w in winners if not np.isnan(w.get("bat_speed", np.nan))]
    winners_4 = [w for w in winners]

    print_attribute_table(winners_4, ATTRS_4, "Past HRD Winners — 4 Core Attributes")
    if winners_5:
        print_attribute_table(winners_5, ATTRS_5, "Past HRD Winners — All 5 Attributes (2024+)")

    # Winner centroids
    centroid_4 = winner_centroid(winners_4, ATTRS_4)
    centroid_5 = winner_centroid(winners_5, ATTRS_5) if winners_5 else None

    print("\nWinner Centroid (average across all available winners):")
    print("-" * 50)
    for a in ATTRS_4:
        print(f"  {a:<24} {centroid_4[a]:.3f}")
    if centroid_5:
        print(f"  {'bat_speed':<24} {centroid_5['bat_speed']:.1f}  (2024+ only)")

    # Current field similarity
    print("\nFetching current field data...")
    field = load_current_field_attributes()

    scored_4 = similarity_scores(field, centroid_4, ATTRS_4)

    print("\nCurrent Field — Similarity to Historical Winner Profile")
    print("-" * 55)
    print(f"  {'Player':<24} {'Similarity Score':>18}")
    print("-" * 55)
    for p in scored_4:
        bar = "#" * int(p["similarity"] / 5)
        print(f"  {p['name']:<24} {p['similarity']:>10.1f} / 100   {bar}")

    if centroid_5 and winners_5:
        scored_5 = similarity_scores(field, centroid_5, ATTRS_5)
        print("\nCurrent Field — Similarity (5 attributes, 2024/25 winners only)")
        print("-" * 55)
        for p in scored_5:
            bar = "#" * int(p["similarity"] / 5)
            print(f"  {p['name']:<24} {p['similarity']:>10.1f} / 100   {bar}")

    if args.suggest_weights:
        print("\nSuggested Weights Based on Winner Z-Scores:")
        print("-" * 50)
        weights, z_scores, field_means = suggest_weights(winners_4, ATTRS_4)
        for a in sorted(z_scores, key=lambda x: -z_scores[x]):
            winner_mean = np.nanmean([w[a] for w in winners_4 if not np.isnan(w.get(a, np.nan))])
            print(f"  {a:<24}  winner avg: {winner_mean:.3f}  field avg: {field_means[a]:.3f}"
                  f"  z: {z_scores[a]:.2f}  -> suggested weight: {weights[a]:.0%}")
        print("\n  Note: bat_speed excluded (insufficient historical data)")
        print("  Redistribute bat_speed weight proportionally once more years available.")


if __name__ == "__main__":
    main()
