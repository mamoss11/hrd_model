# ─────────────────────────────────────────────────────────────
#  HRD Model — Parameter Calibration
#
#  Tunes HR_PROB_MIN, HR_PROB_MAX, and PERFORMANCE_STD by finding
#  the values that minimise the weighted MSE between simulated
#  expected R1 HRs and actual R1 HRs for all 32 contestants
#  from 2022-2025.
#
#  Uses an analytical expected-value calculation (Poisson approx
#  + scipy CDF) — no brute-force simulation needed, runs fast.
#
#  Usage:
#    python calibration.py
# ─────────────────────────────────────────────────────────────
import io
import sys
import warnings
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import ATTRIBUTE_WEIGHTS, PERFORMANCE_CLIP, DERBY_FORMAT

CACHE_FILE = "regression_cache.csv"

# Year weights (same as regression — discount older years slightly)
YEAR_WEIGHTS = {2022: 0.30, 2023: 0.40, 2024: 1.00, 2025: 1.00}

# Actual R1 HRs — swing-off HRs excluded (base round only)
R1_DATA = [
    # 2022
    {"name": "Kyle Schwarber",    "season": 2022, "r1_hrs": 19},
    {"name": "Albert Pujols",     "season": 2022, "r1_hrs": 19},  # swing-off excl.
    {"name": "Juan Soto",         "season": 2022, "r1_hrs": 18},
    {"name": "Jose Ramirez",      "season": 2022, "r1_hrs": 17},
    {"name": "Corey Seager",      "season": 2022, "r1_hrs": 24},
    {"name": "Julio Rodriguez",   "season": 2022, "r1_hrs": 32},
    {"name": "Pete Alonso",       "season": 2022, "r1_hrs": 20},
    {"name": "Ronald Acuna Jr.",  "season": 2022, "r1_hrs": 19},
    # 2023
    {"name": "Luis Robert Jr.",   "season": 2023, "r1_hrs": 28},
    {"name": "Adley Rutschman",   "season": 2023, "r1_hrs": 27},
    {"name": "Adolis Garcia",     "season": 2023, "r1_hrs": 17},
    {"name": "Randy Arozarena",   "season": 2023, "r1_hrs": 24},
    {"name": "Mookie Betts",      "season": 2023, "r1_hrs": 11},
    {"name": "Vlad Guerrero Jr.", "season": 2023, "r1_hrs": 26},
    {"name": "Pete Alonso",       "season": 2023, "r1_hrs": 21},
    {"name": "Julio Rodriguez",   "season": 2023, "r1_hrs": 41},
    # 2024
    {"name": "Teoscar Hernandez", "season": 2024, "r1_hrs": 19},
    {"name": "Bobby Witt Jr.",    "season": 2024, "r1_hrs": 20},
    {"name": "Alec Bohm",         "season": 2024, "r1_hrs": 21},
    {"name": "Jose Ramirez",      "season": 2024, "r1_hrs": 21},
    {"name": "Adolis Garcia",     "season": 2024, "r1_hrs": 18},
    {"name": "Marcell Ozuna",     "season": 2024, "r1_hrs": 16},
    {"name": "Pete Alonso",       "season": 2024, "r1_hrs": 12},
    {"name": "Gunnar Henderson",  "season": 2024, "r1_hrs": 11},
    # 2025
    {"name": "Cal Raleigh",       "season": 2025, "r1_hrs": 17},  # swing-off excl.
    {"name": "Junior Caminero",   "season": 2025, "r1_hrs": 21},
    {"name": "Oneil Cruz",        "season": 2025, "r1_hrs": 21},
    {"name": "Byron Buxton",      "season": 2025, "r1_hrs": 20},
    {"name": "Brent Rooker",      "season": 2025, "r1_hrs": 17},  # swing-off excl.
    {"name": "James Wood",        "season": 2025, "r1_hrs": 16},
    {"name": "Matt Olson",        "season": 2025, "r1_hrs": 15},
    {"name": "Jazz Chisholm Jr.", "season": 2025, "r1_hrs":  3},
]


# ── Data preparation ──────────────────────────────────────────

def load_and_score() -> pd.DataFrame:
    """
    Load regression_cache.csv, compute normalised power score for each
    contestant within their year's field (replicating player_model.py logic).

    2022/2023 players have no bat_speed in Statcast.  We fill those with
    the 2024/2025 field mean (~74.2 mph), which makes bat_speed contribute
    zero differentiation for those years — effectively excluding it while
    keeping the weight arithmetic consistent.
    """
    cache = pd.read_csv(CACHE_FILE)
    r1_df = pd.DataFrame(R1_DATA)

    attr_keys = list(ATTRIBUTE_WEIGHTS.keys())
    weights   = np.array([ATTRIBUTE_WEIGHTS[k] for k in attr_keys])

    # Impute missing bat_speed with mean of 2024/2025 values
    bs_mean = cache.loc[cache["season"] >= 2024, "bat_speed"].mean()
    cache["bat_speed"] = cache["bat_speed"].fillna(bs_mean)

    rows = []
    for season in [2022, 2023, 2024, 2025]:
        yr_cache = cache[cache["season"] == season].copy()
        yr_r1    = r1_df[r1_df["season"] == season].copy()

        merged = pd.merge(yr_r1,
                          yr_cache[["name", "season"] + attr_keys],
                          on=["name", "season"],
                          how="left")

        X = merged[attr_keys].values.astype(float)

        # Min-max normalise within the 8-player field
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        denom   = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
        X_norm  = (X - col_min) / denom

        power_scores = X_norm @ weights

        # Re-normalise power scores to 0–1 (same as _attach_probabilities)
        ps_min, ps_max = power_scores.min(), power_scores.max()
        ps_denom = (ps_max - ps_min) if ps_max > ps_min else 1.0
        norm_scores = (power_scores - ps_min) / ps_denom

        merged["norm_score"]  = norm_scores
        merged["year_weight"] = YEAR_WEIGHTS[season]
        rows.append(merged)

    df = pd.concat(rows, ignore_index=True)

    missing = df[df["norm_score"].isna()]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} players missing from cache:")
        for _, row in missing.iterrows():
            print(f"    {row['name']} ({row['season']})")
        df = df.dropna(subset=["norm_score"])

    return df


# ── Expected R1 HRs (analytical) ─────────────────────────────

def expected_r1_hrs(hr_prob: float,
                    perf_std: float,
                    n_perf: int = 30_000) -> float:
    """
    Analytically compute E[R1 HRs] by integrating over the performance
    factor distribution.

    Uses the Poisson approximation for binomial swings:
      base_hrs | perf ~ Poisson(40 * eff_prob)

    Expected total HRs = E_perf[ eff_prob * (40 + 7 * P(base >= 20 | perf)) ]
    where eff_prob = clip(hr_prob * perf, 0.05, 0.95)
    and   P(base >= 20) = 1 - Poisson_CDF(19, 40 * eff_prob)
    """
    rng    = np.random.default_rng(42)
    perf   = np.clip(rng.normal(1.0, perf_std, n_perf), *PERFORMANCE_CLIP)
    eff_p  = np.clip(hr_prob * perf, 0.05, 0.95)

    base_lambda = 40.0 * eff_p
    p_bonus     = 1.0 - sp_poisson.cdf(19, base_lambda)   # P(base HRs >= 20)

    return float((eff_p * (40.0 + 7.0 * p_bonus)).mean())


def build_lookup_table(hr_prob_vals: list, perf_std_vals: list) -> dict:
    """Pre-compute E[R1_HRs] for every (hr_prob, perf_std) combination."""
    print(f"  Building lookup table "
          f"({len(hr_prob_vals)} x {len(perf_std_vals)} = "
          f"{len(hr_prob_vals)*len(perf_std_vals)} entries) ...")
    table = {}
    for ps in perf_std_vals:
        for hp in hr_prob_vals:
            table[(round(hp, 3), round(ps, 3))] = expected_r1_hrs(hp, ps)
    print("  Done.")
    return table


# ── Grid search ───────────────────────────────────────────────

def _lookup(table: dict, hr_prob: float, perf_std: float) -> float:
    key = (round(hr_prob, 3), round(perf_std, 3))
    if key in table:
        return table[key]
    # Fallback (should rarely trigger)
    return expected_r1_hrs(hr_prob, perf_std)


def grid_search(df: pd.DataFrame,
                table: dict,
                hr_min_vals: list,
                hr_max_vals: list,
                perf_std_vals: list) -> list:

    actual      = df["r1_hrs"].values.astype(float)
    year_w      = df["year_weight"].values.astype(float)
    norm_scores = df["norm_score"].values.astype(float)

    results = []
    for hr_min, hr_max, perf_std in product(hr_min_vals, hr_max_vals, perf_std_vals):
        if hr_min >= hr_max:
            continue

        hr_probs  = hr_min + (hr_max - hr_min) * norm_scores
        predicted = np.array([_lookup(table, p, perf_std) for p in hr_probs])

        residuals = actual - predicted
        wmse      = (year_w * residuals ** 2).sum() / year_w.sum()
        rmse      = float(np.sqrt(wmse))

        results.append({
            "hr_prob_min": round(hr_min, 3),
            "hr_prob_max": round(hr_max, 3),
            "perf_std":    round(perf_std, 3),
            "rmse":        rmse,
        })

    return sorted(results, key=lambda x: x["rmse"])


# ── Comparison table ──────────────────────────────────────────

def print_comparison(df: pd.DataFrame,
                     table: dict,
                     best: dict,
                     current: dict) -> None:
    norm_scores = df["norm_score"].values.astype(float)
    actual      = df["r1_hrs"].values.astype(float)

    def get_pred(params):
        hp = params["hr_prob_min"] + \
             (params["hr_prob_max"] - params["hr_prob_min"]) * norm_scores
        return np.array([_lookup(table, p, params["perf_std"]) for p in hp])

    pred_best = get_pred(best)
    pred_curr = get_pred(current)

    print(f"\n  {'Name':<24} {'Yr':>4} {'Actual':>7} "
          f"{'Best':>7} {'Curr':>7} {'Err_B':>7} {'Err_C':>7}")
    print(f"  {'-'*64}")
    for i, row in df.iterrows():
        eb = pred_best[i] - row["r1_hrs"]
        ec = pred_curr[i] - row["r1_hrs"]
        print(f"  {row['name']:<24} {int(row['season']):>4} "
              f"{row['r1_hrs']:>7.0f} "
              f"{pred_best[i]:>7.1f} {pred_curr[i]:>7.1f} "
              f"{eb:>+7.1f} {ec:>+7.1f}")

    rmse_b = float(np.sqrt(((actual - pred_best) ** 2).mean()))
    rmse_c = float(np.sqrt(((actual - pred_curr) ** 2).mean()))
    print(f"\n  Unweighted RMSE  — Best: {rmse_b:.2f},  Current: {rmse_c:.2f}")
    print(f"  Mean actual R1   — {actual.mean():.1f} HRs")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\nLoading and scoring historical contestants...")
    df = load_and_score()
    print(f"  {len(df)} contestant-seasons loaded.\n")

    # Lookup table covering full hr_prob range
    hr_prob_vals  = [round(x, 3) for x in np.arange(0.20, 0.91, 0.01)]
    perf_std_vals = [round(x, 3) for x in np.arange(0.10, 0.51, 0.05)]
    table = build_lookup_table(hr_prob_vals, perf_std_vals)

    # Grid search parameters
    hr_min_grid   = [round(x, 2) for x in np.arange(0.25, 0.55, 0.05)]
    hr_max_grid   = [round(x, 2) for x in np.arange(0.50, 0.90, 0.05)]
    perf_std_grid = [round(x, 2) for x in np.arange(0.15, 0.50, 0.05)]

    n_combos = sum(1 for a, b, _ in product(hr_min_grid, hr_max_grid, perf_std_grid)
                   if a < b)
    print(f"\nRunning grid search over {n_combos} parameter combinations...")
    results = grid_search(df, table, hr_min_grid, hr_max_grid, perf_std_grid)

    print(f"\nTop 10 parameter sets (by weighted RMSE):")
    print(f"  {'MIN':>6} {'MAX':>6} {'STD':>6} {'RMSE':>8}")
    print(f"  {'-'*32}")
    for r in results[:10]:
        print(f"  {r['hr_prob_min']:>6.2f} {r['hr_prob_max']:>6.2f} "
              f"{r['perf_std']:>6.2f} {r['rmse']:>8.3f}")

    best    = results[0]
    current = {"hr_prob_min": 0.40, "hr_prob_max": 0.70, "perf_std": 0.32}

    curr_match = next(
        (r for r in results
         if r["hr_prob_min"] == current["hr_prob_min"]
         and r["hr_prob_max"] == current["hr_prob_max"]
         and r["perf_std"]    == current["perf_std"]),
        None
    )
    curr_rmse = f"{curr_match['rmse']:.3f}" if curr_match else "N/A"

    print(f"\nCurrent: MIN={current['hr_prob_min']}, MAX={current['hr_prob_max']}, "
          f"STD={current['perf_std']}  →  RMSE={curr_rmse}")
    print(f"Best:    MIN={best['hr_prob_min']}, MAX={best['hr_prob_max']}, "
          f"STD={best['perf_std']}  →  RMSE={best['rmse']:.3f}")

    print_comparison(df, table, best, current)

    print(f"\nTo apply best params, update config.py:")
    print(f"  HR_PROB_MIN      = {best['hr_prob_min']}")
    print(f"  HR_PROB_MAX      = {best['hr_prob_max']}")
    print(f"  PERFORMANCE_STD  = {best['perf_std']}")


if __name__ == "__main__":
    main()
