# ─────────────────────────────────────────────────────────────
#  HRD Model — 2025 Market Validation
#
#  Runs the model against the actual 2025 field using pre-event
#  Statcast data, then compares model fair probs against closing
#  prices pulled from SPORTSCONTENT.DBO.SELECTIONSHISTORY.
#
#  Usage:  python validate_2025.py
# ─────────────────────────────────────────────────────────────
import sys
import time
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import simulator as sim_module
from data_fetcher import load_player_attributes
from player_model import build_player_models
from simulator import run_simulations
from markets import price_all_markets
from config import N_SIMULATIONS


# ── 2025 actual field ──────────────────────────────────────────
FIELD_2025 = {
    "venue":        "Truist Park",
    "park":         {"hr_factor_R": 1.04, "hr_factor_L": 1.06, "dist_bonus_ft": 0.0},
    "season_start": "2025-03-27",
    "season_end":   "2025-07-13",   # use pre-Derby stats only
    "winner":       "Cal Raleigh",
    "players": [
        {"name": "Cal Raleigh",       "mlb_id": 663728, "team": "SEA", "league": "AL", "nationality": "USA",           "bats": "L"},
        {"name": "Oneil Cruz",        "mlb_id": 665833, "team": "PIT", "league": "NL", "nationality": "International", "bats": "L"},
        {"name": "James Wood",        "mlb_id": 695578, "team": "WSH", "league": "NL", "nationality": "USA",           "bats": "L"},
        {"name": "Byron Buxton",      "mlb_id": 621439, "team": "MIN", "league": "AL", "nationality": "USA",           "bats": "R"},
        {"name": "Matt Olson",        "mlb_id": 621566, "team": "ATL", "league": "NL", "nationality": "USA",           "bats": "L"},
        {"name": "Brent Rooker",      "mlb_id": 667670, "team": "OAK", "league": "AL", "nationality": "USA",           "bats": "R"},
        {"name": "Junior Caminero",   "mlb_id": 691406, "team": "TB",  "league": "AL", "nationality": "International", "bats": "R"},
        {"name": "Jazz Chisholm Jr.", "mlb_id": 665862, "team": "NYY", "league": "AL", "nationality": "USA",           "bats": "L"},
    ],
}

# ── Market closing lines from Snowflake (SELECTIONSHISTORY) ───
# Prices filtered: MSGSOURCETRANSFERDATE < 2025-07-14 23:00:00
# Decimal odds; fair prob = (1/decimal) / sum(1/decimal) for each market.
#
# Winner (eventid 260952723) — raw decimal odds:
MARKET_WINNER_DECIMAL = {
    "Cal Raleigh":       4.2,
    "Oneil Cruz":        4.5,
    "James Wood":        4.9,
    "Byron Buxton":      9.0,
    "Matt Olson":        9.0,
    "Brent Rooker":      9.5,
    "Junior Caminero":  11.0,
    "Jazz Chisholm Jr.":15.0,
}

# Make the Final (eventid 261494866) — raw decimal odds:
MARKET_FINAL_DECIMAL = {
    "Cal Raleigh":       2.30,
    "Oneil Cruz":        2.50,
    "James Wood":        3.20,
    "Matt Olson":        4.10,
    "Brent Rooker":      4.70,
    "Byron Buxton":      4.80,
    "Jazz Chisholm Jr.": 5.75,
    "Junior Caminero":   6.00,
}

# Totals — market mid-point lines and fair Over% (vig-removed)
# Total HRs in Derby (261676380):  O/U 239.5 @ 1.91/1.91  → fair 50.0% each side
TOTAL_DERBY_LINE    = 239.5
TOTAL_DERBY_OVER_FAIR = 0.500   # 1.91/1.91 is near-even; vig-removed ~50%

# Total HRs in Round 1 (261677172): O/U 154.5 @ 2.0/1.83 (+100/-120)
# Over implied 50.0%, Under 54.6%; sum 104.6% → fair Over = 47.8%
TOTAL_R1_LINE       = 154.5
TOTAL_R1_OVER_FAIR  = 0.478

# Most HRs by any player in R1 (261677503): O/U 25.5 @ 1.95/1.87 (+95/-115)
# Over implied 51.3%, Under 53.5%; sum 104.8% → fair Over = 48.9%
MOST_R1_LINE        = 25.5
MOST_R1_OVER_FAIR   = 0.489


# ── Helpers ───────────────────────────────────────────────────

def _vig_remove(decimal_map: dict, n_winners: int = 1) -> dict:
    """Convert raw decimal odds to vig-removed fair probs.

    n_winners: how many outcomes win (1 for outright winner, 2 for Make the Final, etc.)
    The vig-free probs are scaled so they sum to n_winners.
    """
    implied = {k: 1.0 / v for k, v in decimal_map.items()}
    total   = sum(implied.values())
    # normalize so that the sum equals n_winners (fair market structure)
    scale   = n_winners / total
    return {k: v * scale for k, v in implied.items()}


def _american(p: float) -> str:
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    else:
        return f"+{round((1 - p) / p * 100)}"


def _diff_label(model_p: float, market_p: float) -> str:
    """+ means model is higher than market, - means model is lower."""
    d = (model_p - market_p) * 100
    return f"{d:+.1f}pp"


# ── Main ──────────────────────────────────────────────────────

def main():
    cfg = FIELD_2025

    print(f"\n{'=' * 68}")
    print(f"  2025 HRD Model vs Market Closing Lines")
    print(f"  Venue: {cfg['venue']}  |  Park R={cfg['park']['hr_factor_R']:.2f}  L={cfg['park']['hr_factor_L']:.2f}")
    print(f"  Season data: {cfg['season_start']} to {cfg['season_end']}")
    print(f"  Sims: {N_SIMULATIONS:,}")
    print(f"{'=' * 68}")

    # 1. Apply park factors
    sim_module._PARK_HR_FACTOR_R = cfg["park"]["hr_factor_R"]
    sim_module._PARK_HR_FACTOR_L = cfg["park"]["hr_factor_L"]
    sim_module._PARK_DIST_BONUS  = cfg["park"]["dist_bonus_ft"]

    # 2. Fetch Statcast
    print("\nFetching Statcast data...", flush=True)
    players = load_player_attributes(
        cfg["players"],
        start=cfg["season_start"],
        end=cfg["season_end"],
    )

    # 3. Build models
    players = build_player_models(players)

    # ── Raw attribute table ────────────────────────────────────
    from config import ATTRIBUTE_WEIGHTS
    attr_keys = list(ATTRIBUTE_WEIGHTS.keys())
    weights   = [ATTRIBUTE_WEIGHTS[k] for k in attr_keys]

    import numpy as np
    raw = np.array([[p[k] for k in attr_keys] for p in players], dtype=float)
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    normed  = (raw - col_min) / denom
    weighted = normed * weights

    short = {
        "bat_speed":         "BatSpd",
        "max_exit_velo":     "MaxEV",
        "pulled_barrel_pct": "PullBrl%",
        "hr_per_pa":         "HR/PA",
        "pct90_exit_velo":   "EV90",
    }
    order = sorted(range(len(players)), key=lambda i: -players[i]["power_score"])

    print(f"\n{'─' * 68}")
    print("  RAW STATCAST ATTRIBUTES  (sorted by power score)")
    print(f"{'─' * 68}")
    hdr = f"  {'Player':<23}"
    for k in attr_keys:
        hdr += f"  {short[k]:>9}"
    print(hdr)
    wdr = f"  {'weight →':<23}"
    for w in weights:
        wdr += f"  {f'({int(w*100)}%)':>9}"
    print(wdr)
    print("  " + "-" * 66)
    for i in order:
        p   = players[i]
        row = f"  {p['name']:<23}"
        for j, k in enumerate(attr_keys):
            v = raw[i, j]
            if k in ("pulled_barrel_pct", "hr_per_pa"):
                row += f"  {v:>9.4f}"
            else:
                row += f"  {v:>9.1f}"
        print(row)

    print(f"\n{'─' * 68}")
    print("  NORMALIZED WEIGHTED CONTRIBUTIONS  (0–1 per attr × weight)")
    print(f"{'─' * 68}")
    hdr2 = f"  {'Player':<23}"
    for k in attr_keys:
        hdr2 += f"  {short[k]:>9}"
    hdr2 += f"  {'Total':>9}"
    print(hdr2)
    print("  " + "-" * 66)
    for i in order:
        p   = players[i]
        row = f"  {p['name']:<23}"
        for j in range(len(attr_keys)):
            row += f"  {weighted[i, j]:>9.4f}"
        row += f"  {weighted[i].sum():>9.4f}"
        print(row)

    print(f"\n{'─' * 68}")
    print("  SUMMARY")
    print(f"{'─' * 68}")
    print(f"  {'Player':<23} {'Power':>7} {'HR%/sw':>8} {'MeanDist':>9}")
    print("  " + "-" * 52)
    for p in sorted(players, key=lambda x: -x["power_score"]):
        print(f"  {p['name']:<23} {p['power_score']:>7.4f} {p['hr_prob']:>7.1%} {p['mean_hr_dist']:>8.1f}ft")

    # 4. Simulate
    print(f"\nRunning {N_SIMULATIONS:,} simulations...", flush=True)
    t0 = time.time()
    sims = run_simulations(players, n=N_SIMULATIONS)
    print(f"Done in {time.time()-t0:.1f}s.  Pricing...", flush=True)

    # 5. Price
    markets = price_all_markets(players, sims)

    # ── Compute market fair probs ──────────────────────────────
    winner_fair = _vig_remove(MARKET_WINNER_DECIMAL, n_winners=1)
    final_fair  = _vig_remove(MARKET_FINAL_DECIMAL,  n_winners=2)  # 2 finalists advance

    names = [p["name"] for p in players]

    # ── Comparison 1: WINNER ───────────────────────────────────
    print(f"\n{'─' * 68}")
    print("  WINNER  (EventID 260952723 | pre-event closing)")
    print(f"{'─' * 68}")
    print(f"  {'Player':<23} {'Market':>9} {'Model':>9} {'Diff':>9}  {'Mkt Odds':>9}  {'Mdl Odds':>9}")
    print("  " + "-" * 65)

    model_winner = markets["Winner"]
    for name in sorted(names, key=lambda n: -winner_fair.get(n, 0)):
        mkt_p   = winner_fair.get(name, 0)
        mdl_row = model_winner.get(name, {})
        mdl_p   = mdl_row.get("prob", 0)
        print(
            f"  {name:<23} {mkt_p:>8.1%} {mdl_p:>8.1%} {_diff_label(mdl_p, mkt_p):>9}"
            f"  {_american(mkt_p):>9}  {_american(mdl_p):>9}"
        )

    # ── Comparison 2: MAKE THE FINAL ──────────────────────────
    print(f"\n{'─' * 68}")
    print("  MAKE THE FINAL  (EventID 261494866 | pre-event closing)")
    print(f"{'─' * 68}")
    print(f"  {'Player':<23} {'Market':>9} {'Model':>9} {'Diff':>9}  {'Mkt Odds':>9}  {'Mdl Odds':>9}")
    print("  " + "-" * 65)

    model_final = markets["Make the Final"]
    for name in sorted(names, key=lambda n: -final_fair.get(n, 0)):
        mkt_p   = final_fair.get(name, 0)
        mdl_row = model_final.get(name, {})
        mdl_p   = mdl_row.get("prob", 0)
        print(
            f"  {name:<23} {mkt_p:>8.1%} {mdl_p:>8.1%} {_diff_label(mdl_p, mkt_p):>9}"
            f"  {_american(mkt_p):>9}  {_american(mdl_p):>9}"
        )

    # ── Comparison 3: TOTALS ───────────────────────────────────
    print(f"\n{'─' * 68}")
    print("  TOTALS  (pre-event closing lines)")
    print(f"{'─' * 68}")
    print(f"  {'Market':<35} {'Mkt Line':>9} {'Mkt Over%':>10} {'Mdl Over%':>10} {'Diff':>9}")
    print("  " + "-" * 65)

    def _show_total(label, market_key, market_line, market_over_fair):
        if market_key not in markets:
            print(f"  {label:<35} {'N/A':>9}")
            return
        ou = markets[market_key]
        over_key = f"Over {market_line}"
        mdl_over = ou.get(over_key, {}).get("prob", None)

        # find nearest line in model if exact line not generated
        if mdl_over is None:
            candidates = {k: v for k, v in ou.items() if k.startswith("Over ")}
            if candidates:
                nearest = min(candidates.keys(),
                              key=lambda k: abs(float(k.split()[1]) - market_line))
                actual_line  = float(nearest.split()[1])
                mdl_over     = candidates[nearest]["prob"]
                note = f"  [model line: {actual_line}]"
            else:
                print(f"  {label:<35} no model lines found")
                return
        else:
            note = ""

        diff = _diff_label(mdl_over, market_over_fair)
        print(
            f"  {label:<35} {market_line:>9.1f} {market_over_fair:>9.1%} {mdl_over:>9.1%} {diff:>9}{note}"
        )

    _show_total("Total HRs in Derby",          "Total HRs",                    TOTAL_DERBY_LINE,  TOTAL_DERBY_OVER_FAIR)
    _show_total("Total HRs in Round 1",        "Total HRs in Round 1",         TOTAL_R1_LINE,     TOTAL_R1_OVER_FAIR)
    _show_total("Most HRs by Any Player in R1","Most HRs by Any Player in Round 1", MOST_R1_LINE,      MOST_R1_OVER_FAIR)

    # ── Model median totals ────────────────────────────────────
    import numpy as np
    total_derby_vals = [s["total_derby_hrs"] for s in sims]
    total_r1_vals    = [s["total_r1_hrs"]    for s in sims]
    most_r1_vals     = [s["most_r1_hrs_count"] for s in sims]

    print(f"\n{'─' * 68}")
    print("  MODEL TOTAL DISTRIBUTIONS")
    print(f"{'─' * 68}")
    print(f"  {'Market':<35} {'Mean':>8} {'Median':>8} {'P10':>8} {'P90':>8}")
    print("  " + "-" * 65)
    for label, vals in [
        ("Total HRs in Derby",           total_derby_vals),
        ("Total HRs in Round 1",         total_r1_vals),
        ("Most HRs by Any Player in R1", most_r1_vals),
    ]:
        arr = np.array(vals)
        print(
            f"  {label:<35} {arr.mean():>8.1f} {np.median(arr):>8.1f}"
            f" {np.percentile(arr, 10):>8.1f} {np.percentile(arr, 90):>8.1f}"
        )

    # ── Winner ────────────────────────────────────────────────
    actual = cfg["winner"]
    ranked = sorted(model_winner.items(), key=lambda x: x[1]["prob"], reverse=True)
    rank   = next((i + 1 for i, (n, _) in enumerate(ranked) if n == actual), None)
    print(f"\n  *** Actual 2025 winner: {actual}  (model rank: #{rank}) ***\n")


if __name__ == "__main__":
    main()
