# ─────────────────────────────────────────────────────────────
#  HRD Model — Regression Analysis
#
#  Fetches Statcast attributes for all HRD contestants 2022-2025,
#  then runs weighted OLS regression to find which attributes best
#  predict avg HRs per round. Outputs suggested weights for config.py.
#
#  Usage:
#    python regression_analysis.py              # fetch from Statcast API
#    python regression_analysis.py --no-fetch   # load cached CSV (faster)
# ─────────────────────────────────────────────────────────────
import io
import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_fetcher import fetch_statcast, compute_attributes

CACHE_FILE = "regression_cache.csv"

# ── Season windows ────────────────────────────────────────────
SEASON_WINDOWS = {
    2022: ("2022-04-07", "2022-07-11"),
    2023: ("2023-03-30", "2023-07-10"),
    2024: ("2024-03-28", "2024-07-13"),
    2025: ("2025-03-27", "2025-07-14"),
}

# ── Year weights (older years discounted, format change in 2024) ──
YEAR_WEIGHTS = {
    2022: 0.30,
    2023: 0.40,
    2024: 1.00,
    2025: 1.00,
}

# ── All contestants 2022-2025 ─────────────────────────────────
# avg_hrs: average HRs per round
#   - swing-off HRs excluded
#   - sandbagged rounds excluded (score < 50% of R1 → dropped)
# bats: switch hitters defaulted to R (natural/dominant side)
CONTESTANTS = [
    # ── 2022 ──────────────────────────────────────────────────
    {"name": "Kyle Schwarber",    "mlb_id": 656941, "bats": "L", "season": 2022, "avg_hrs": 19.00},
    {"name": "Albert Pujols",     "mlb_id": 405395, "bats": "R", "season": 2022, "avg_hrs": 17.00},
    {"name": "Juan Soto",         "mlb_id": 665742, "bats": "L", "season": 2022, "avg_hrs": 17.67},
    {"name": "Jose Ramirez",      "mlb_id": 608070, "bats": "R", "season": 2022, "avg_hrs": 17.00},  # switch -> R
    {"name": "Corey Seager",      "mlb_id": 608369, "bats": "L", "season": 2022, "avg_hrs": 24.00},
    {"name": "Julio Rodriguez",   "mlb_id": 677594, "bats": "R", "season": 2022, "avg_hrs": 27.00},
    {"name": "Pete Alonso",       "mlb_id": 624413, "bats": "R", "season": 2022, "avg_hrs": 21.50},
    {"name": "Ronald Acuna Jr.",  "mlb_id": 660670, "bats": "R", "season": 2022, "avg_hrs": 19.00},
    # ── 2023 ──────────────────────────────────────────────────
    {"name": "Luis Robert Jr.",   "mlb_id": 673357, "bats": "R", "season": 2023, "avg_hrs": 25.00},
    {"name": "Adley Rutschman",   "mlb_id": 668939, "bats": "R", "season": 2023, "avg_hrs": 27.00},  # switch -> R
    {"name": "Adolis Garcia",     "mlb_id": 666969, "bats": "R", "season": 2023, "avg_hrs": 17.00},
    {"name": "Randy Arozarena",   "mlb_id": 668227, "bats": "R", "season": 2023, "avg_hrs": 27.33},
    {"name": "Mookie Betts",      "mlb_id": 605141, "bats": "R", "season": 2023, "avg_hrs": 11.00},
    {"name": "Vlad Guerrero Jr.", "mlb_id": 665489, "bats": "R", "season": 2023, "avg_hrs": 24.00},
    {"name": "Pete Alonso",       "mlb_id": 624413, "bats": "R", "season": 2023, "avg_hrs": 21.00},
    {"name": "Julio Rodriguez",   "mlb_id": 677594, "bats": "R", "season": 2023, "avg_hrs": 41.00},
    # ── 2024 ──────────────────────────────────────────────────
    {"name": "Teoscar Hernandez", "mlb_id": 606192, "bats": "R", "season": 2024, "avg_hrs": 15.67},
    {"name": "Bobby Witt Jr.",    "mlb_id": 677951, "bats": "R", "season": 2024, "avg_hrs": 16.67},
    {"name": "Alec Bohm",         "mlb_id": 664761, "bats": "R", "season": 2024, "avg_hrs": 17.50},
    {"name": "Jose Ramirez",      "mlb_id": 608070, "bats": "R", "season": 2024, "avg_hrs": 16.50},  # switch -> R
    {"name": "Adolis Garcia",     "mlb_id": 666969, "bats": "R", "season": 2024, "avg_hrs": 18.00},
    {"name": "Marcell Ozuna",     "mlb_id": 542303, "bats": "R", "season": 2024, "avg_hrs": 16.00},
    {"name": "Pete Alonso",       "mlb_id": 624413, "bats": "R", "season": 2024, "avg_hrs": 12.00},
    {"name": "Gunnar Henderson",  "mlb_id": 683002, "bats": "L", "season": 2024, "avg_hrs": 11.00},
    # ── 2025 ──────────────────────────────────────────────────
    {"name": "Cal Raleigh",       "mlb_id": 663728, "bats": "L", "season": 2025, "avg_hrs": 18.00},
    {"name": "Junior Caminero",   "mlb_id": 691406, "bats": "R", "season": 2025, "avg_hrs": 18.00},
    {"name": "Oneil Cruz",        "mlb_id": 665833, "bats": "L", "season": 2025, "avg_hrs": 17.00},
    {"name": "Byron Buxton",      "mlb_id": 621439, "bats": "R", "season": 2025, "avg_hrs": 20.00},
    {"name": "Brent Rooker",      "mlb_id": 667670, "bats": "R", "season": 2025, "avg_hrs": 17.00},
    {"name": "James Wood",        "mlb_id": 695578, "bats": "L", "season": 2025, "avg_hrs": 16.00},
    {"name": "Matt Olson",        "mlb_id": 621566, "bats": "L", "season": 2025, "avg_hrs": 15.00},
    {"name": "Jazz Chisholm Jr.", "mlb_id": 665862, "bats": "L", "season": 2025, "avg_hrs":  3.00},
]

ATTRS_4 = ["max_exit_velo", "pct90_exit_velo", "avg_exit_velo", "pulled_barrel_pct",
           "total_barrel_rate", "hr_per_pa", "mean_launch_angle", "ideal_la_pct"]
ATTRS_5 = ["bat_speed"] + ATTRS_4


# ── Data fetching ─────────────────────────────────────────────

def fetch_all_attributes(contestants: list) -> pd.DataFrame:
    rows = []
    for c in contestants:
        start, end = SEASON_WINDOWS[c["season"]]
        print(f"  Fetching {c['name']} ({c['season']}) ...")
        try:
            df    = fetch_statcast(c["mlb_id"], start, end)
            attrs = compute_attributes(df, c["bats"])
        except Exception as exc:
            print(f"    WARNING: {exc}")
            attrs = {k: np.nan for k in ATTRS_5}
            attrs["total_balls_hit"] = 0
        rows.append({
            "name":    c["name"],
            "mlb_id":  c["mlb_id"],
            "season":  c["season"],
            "avg_hrs": c["avg_hrs"],
            **attrs,
        })
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n  Cached to {CACHE_FILE}")
    return df


def load_cached() -> pd.DataFrame:
    df = pd.read_csv(CACHE_FILE)
    print(f"  Loaded {len(df)} rows from {CACHE_FILE}")
    return df


# ── Ridge Regression ──────────────────────────────────────────

def _ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Weighted Ridge: minimises Σ w_i(y_i - x_i^T β)² + λ‖β‖². No intercept (centre y first)."""
    p = X.shape[1]
    XtWX = (X.T * w) @ X
    XtWy = X.T @ (w * y)
    return np.linalg.solve(XtWX + lam * np.eye(p), XtWy)


def _loo_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Weighted leave-one-out CV R² for a given lambda."""
    n = len(y)
    loo_preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        loo_preds[i] = X[i] @ _ridge_fit(X[mask], y[mask], w[mask], lam)
    w_mean = (w * y).sum() / w.sum()
    ss_res = (w * (y - loo_preds) ** 2).sum()
    ss_tot = (w * (y - w_mean) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def run_regression(df: pd.DataFrame, attrs: list, label: str) -> dict:
    """
    Weighted Ridge regression on z-scored features with LOO-CV lambda selection.
    Returns normalised positive coefficients as suggested weights.
    """
    subset = df.dropna(subset=attrs + ["avg_hrs"]).copy()
    n = len(subset)
    if n < len(attrs) + 2:
        print(f"\n{label}: insufficient data ({n} rows), skipping.")
        return {}

    subset["w"] = subset["season"].map(YEAR_WEIGHTS).fillna(1.0)

    X = subset[attrs].values.astype(float)
    y = subset["avg_hrs"].values.astype(float)
    w = subset["w"].values.astype(float)

    # Weighted z-score standardisation
    w_sum  = w.sum()
    x_mean = (w[:, None] * X).sum(axis=0) / w_sum
    x_std  = np.sqrt((w[:, None] * (X - x_mean) ** 2).sum(axis=0) / w_sum)
    x_std  = np.where(x_std == 0, 1.0, x_std)
    X_std  = (X - x_mean) / x_std

    # Centre y (intercept = weighted mean of y)
    y_mean = (w * y).sum() / w_sum
    y_c    = y - y_mean

    # LOO-CV to select lambda
    lambdas   = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    loo_scores = {lam: _loo_r2(X_std, y_c, w, lam) for lam in lambdas}
    best_lam   = max(loo_scores, key=loo_scores.get)

    # Final fit with best lambda
    beta   = _ridge_fit(X_std, y_c, w, best_lam)
    y_pred = X_std @ beta

    # In-sample weighted R²
    ss_res = (w * (y_c - y_pred) ** 2).sum()
    ss_tot = (w * y_c ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Normalise positive coefficients to sum to 1
    pos     = np.maximum(beta, 0.0)
    total   = pos.sum()
    weights = pos / total if total > 0 else np.ones(len(attrs)) / len(attrs)

    print(f"\n{label}")
    print(f"  n={n}, λ={best_lam} (LOO-CV R²={loo_scores[best_lam]:.3f}), in-sample R²={r2:.3f}")
    print(f"  {'Attribute':<24} {'Std Coef':>10}  {'Weight':>8}")
    print(f"  {'-'*48}")
    for a, b, wt in zip(attrs, beta, weights):
        neg_flag = "  <- negative, zeroed" if b < 0 else ""
        print(f"  {a:<24} {b:>10.3f}  {wt:>7.1%}{neg_flag}")

    return dict(zip(attrs, weights))


# ── Random Forest Feature Importances ─────────────────────────

def run_random_forest(df: pd.DataFrame, attrs: list, label: str) -> dict:
    """
    Random Forest regressor — uses MDI feature importances.
    Handles multicollinearity and non-linear relationships better than Ridge.
    Year weights applied via sample_weight.
    """
    if not _SKLEARN:
        print(f"\n{label}: sklearn not installed. Run: pip install scikit-learn")
        return {}

    subset = df.dropna(subset=attrs + ["avg_hrs"]).copy()
    n = len(subset)
    if n < len(attrs) + 2:
        print(f"\n{label}: insufficient data ({n} rows), skipping.")
        return {}

    subset["w"] = subset["season"].map(YEAR_WEIGHTS).fillna(1.0)

    X = subset[attrs].values.astype(float)
    y = subset["avg_hrs"].values.astype(float)
    w = subset["w"].values.astype(float)

    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y, sample_weight=w)

    importances = rf.feature_importances_
    total = importances.sum()
    weights = importances / total if total > 0 else np.ones(len(attrs)) / len(attrs)

    r2 = rf.score(X, y, sample_weight=w)

    print(f"\n{label}")
    print(f"  n={n}, n_estimators=500, in-sample R²={r2:.3f}")
    print(f"  {'Attribute':<24} {'Importance':>12}  {'Weight':>8}")
    print(f"  {'-'*50}")
    order = np.argsort(importances)[::-1]
    for i in order:
        print(f"  {attrs[i]:<24} {importances[i]:>12.4f}  {weights[i]:>7.1%}")

    return dict(zip(attrs, weights))


# ── Output ────────────────────────────────────────────────────

def print_suggested_config(weights_4: dict, weights_5: dict,
                           rf_weights_4: dict, rf_weights_5: dict) -> None:
    print("\n" + "=" * 55)
    print("SUGGESTED WEIGHTS FOR config.py")
    print("=" * 55)

    from config import ATTRIBUTE_WEIGHTS
    bs_share = ATTRIBUTE_WEIGHTS.get("bat_speed", 0.50)
    scale    = 1.0 - bs_share
    CONFIG_ATTRS = ["max_exit_velo", "pct90_exit_velo", "pulled_barrel_pct", "hr_per_pa"]

    if weights_4:
        print("\n  Ridge (bat_speed excluded, 2022-2025):")
        print(f"  (bat_speed held at {bs_share:.0%}, remaining {scale:.0%} redistributed from config attrs only)")
        config_wts = {a: weights_4.get(a, 0.0) for a in CONFIG_ATTRS}
        total = sum(config_wts.values())
        for a, wt in config_wts.items():
            norm = (wt / total * scale) if total > 0 else scale / len(CONFIG_ATTRS)
            print(f"    \"{a}\": {norm:.3f},")
        print(f"    \"bat_speed\": {bs_share:.3f},")

    if rf_weights_4:
        print("\n  Random Forest (bat_speed excluded, 2022-2025):")
        print(f"  (bat_speed held at {bs_share:.0%}, remaining {scale:.0%} redistributed from config attrs only)")
        config_wts = {a: rf_weights_4.get(a, 0.0) for a in CONFIG_ATTRS}
        total = sum(config_wts.values())
        for a, wt in config_wts.items():
            norm = (wt / total * scale) if total > 0 else scale / len(CONFIG_ATTRS)
            print(f"    \"{a}\": {norm:.3f},")
        print(f"    \"bat_speed\": {bs_share:.3f},")

    if weights_5:
        print("\n  Ridge (2024-2025 only, includes bat_speed):")
        config_wts = {a: weights_5.get(a, 0.0) for a in CONFIG_ATTRS + ["bat_speed"]}
        total = sum(config_wts.values())
        for a, wt in config_wts.items():
            norm = (wt / total) if total > 0 else 1.0 / len(config_wts)
            print(f"    \"{a}\": {norm:.3f},")

    if rf_weights_5:
        print("\n  Random Forest (2024-2025 only, includes bat_speed):")
        config_wts = {a: rf_weights_5.get(a, 0.0) for a in CONFIG_ATTRS + ["bat_speed"]}
        total = sum(config_wts.values())
        for a, wt in config_wts.items():
            norm = (wt / total) if total > 0 else 1.0 / len(config_wts)
            print(f"    \"{a}\": {norm:.3f},")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API calls and load from cached CSV")
    args = parser.parse_args()

    if args.no_fetch:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: {CACHE_FILE} not found. Run without --no-fetch first.")
            sys.exit(1)
        print("\nLoading cached data...")
        df = load_cached()
    else:
        print(f"\nFetching Statcast data for {len(CONTESTANTS)} contestant-seasons...")
        df = fetch_all_attributes(CONTESTANTS)

    print(f"\n{'Name':<24} {'Yr':>4} {'Avg':>6} {'MaxEV':>6} {'EV90':>6} {'AvgEV':>6} "
          f"{'PBrl':>6} {'TBrl':>6} {'HR/PA':>6} {'LA':>6} {'IdLA%':>6} {'BatSpd':>7}")
    print("-" * 102)
    for _, row in df.iterrows():
        bs = f"{row['bat_speed']:>7.1f}" if pd.notna(row.get("bat_speed")) else f"{'N/A':>7}"
        print(f"  {row['name']:<22} {int(row['season']):>4} {row['avg_hrs']:>6.1f} "
              f"{row['max_exit_velo']:>6.1f} {row['pct90_exit_velo']:>6.1f} {row['avg_exit_velo']:>6.1f} "
              f"{row['pulled_barrel_pct']:>6.3f} {row['total_barrel_rate']:>6.3f} "
              f"{row['hr_per_pa']:>6.3f} "
              f"{row['mean_launch_angle']:>6.1f} {row['ideal_la_pct']:>6.3f}{bs}")

    # Ridge regression: all years (2022-2025)
    w4 = run_regression(df, ATTRS_4, "Ridge Regression (bat_speed excluded, 2022-2025)")

    # Ridge regression: 2024+2025 only (bat_speed available)
    df_recent = df[df["season"] >= 2024].copy()
    w5 = run_regression(df_recent, ATTRS_5, "Ridge Regression (2024-2025, includes bat_speed)")

    # Random Forest: all years (2022-2025)
    rf4 = run_random_forest(df, ATTRS_4, "Random Forest (bat_speed excluded, 2022-2025)")

    # Random Forest: 2024+2025 only (bat_speed available)
    rf5 = run_random_forest(df_recent, ATTRS_5, "Random Forest (2024-2025, includes bat_speed)")

    print_suggested_config(w4, w5, rf4, rf5)


if __name__ == "__main__":
    main()
