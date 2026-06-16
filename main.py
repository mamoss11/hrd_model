# ─────────────────────────────────────────────────────────────
#  HRD Model — Entry Point
#
#  Usage:
#    python main.py                  # uses PLAYERS from config.py
#    python main.py --demo           # runs with placeholder test players
#    python main.py --manual         # skip Statcast fetch; enter stats manually
#    python main.py --output out.json  # also save results to JSON
# ─────────────────────────────────────────────────────────────
import argparse
import io
import sys
import numpy as np

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import PLAYERS, N_SIMULATIONS, HRD_VENUE, PARK_FACTORS
from data_fetcher import load_player_attributes
from player_model import build_player_models
from simulator import run_simulations
from markets import price_all_markets
from formatter import print_markets, save_markets_json


# ── demo player set (placeholder stats — replace with real data) ──
DEMO_PLAYERS = [
    {"name": "Player A", "mlb_id": None, "team": "NYY", "league": "AL",
     "nationality": "USA",           "bats": "R",
     "bat_speed": 77.5, "max_exit_velo": 119.4, "pct90_exit_velo": 109.5,
     "pulled_barrel_pct": 0.095, "hr_per_pa": 0.065, "avg_exit_velo": 94.1,
     "total_barrel_rate": 0.12, "mean_launch_angle": 14.2, "total_balls_hit": 320},

    {"name": "Player B", "mlb_id": None, "team": "LAD", "league": "NL",
     "nationality": "International", "bats": "R",
     "bat_speed": 76.1, "max_exit_velo": 117.8, "pct90_exit_velo": 108.0,
     "pulled_barrel_pct": 0.082, "hr_per_pa": 0.058, "avg_exit_velo": 92.8,
     "total_barrel_rate": 0.10, "mean_launch_angle": 13.5, "total_balls_hit": 290},

    {"name": "Player C", "mlb_id": None, "team": "ATL", "league": "NL",
     "nationality": "USA",           "bats": "R",
     "bat_speed": 74.8, "max_exit_velo": 116.2, "pct90_exit_velo": 106.5,
     "pulled_barrel_pct": 0.078, "hr_per_pa": 0.052, "avg_exit_velo": 91.5,
     "total_barrel_rate": 0.09, "mean_launch_angle": 12.8, "total_balls_hit": 310},

    {"name": "Player D", "mlb_id": None, "team": "HOU", "league": "AL",
     "nationality": "USA",           "bats": "R",
     "bat_speed": 73.5, "max_exit_velo": 115.0, "pct90_exit_velo": 105.0,
     "pulled_barrel_pct": 0.071, "hr_per_pa": 0.048, "avg_exit_velo": 90.2,
     "total_barrel_rate": 0.08, "mean_launch_angle": 12.1, "total_balls_hit": 280},

    {"name": "Player E", "mlb_id": None, "team": "NYM", "league": "NL",
     "nationality": "International", "bats": "L",
     "bat_speed": 75.2, "max_exit_velo": 116.9, "pct90_exit_velo": 107.0,
     "pulled_barrel_pct": 0.086, "hr_per_pa": 0.055, "avg_exit_velo": 92.0,
     "total_barrel_rate": 0.11, "mean_launch_angle": 13.9, "total_balls_hit": 300},

    {"name": "Player F", "mlb_id": None, "team": "MIL", "league": "NL",
     "nationality": "USA",           "bats": "R",
     "bat_speed": 72.9, "max_exit_velo": 114.5, "pct90_exit_velo": 104.0,
     "pulled_barrel_pct": 0.065, "hr_per_pa": 0.044, "avg_exit_velo": 89.8,
     "total_barrel_rate": 0.07, "mean_launch_angle": 11.5, "total_balls_hit": 265},

    {"name": "Player G", "mlb_id": None, "team": "SEA", "league": "AL",
     "nationality": "International", "bats": "R",
     "bat_speed": 74.1, "max_exit_velo": 115.8, "pct90_exit_velo": 106.0,
     "pulled_barrel_pct": 0.074, "hr_per_pa": 0.050, "avg_exit_velo": 91.0,
     "total_barrel_rate": 0.09, "mean_launch_angle": 13.0, "total_balls_hit": 295},

    {"name": "Player H", "mlb_id": None, "team": "TOR", "league": "AL",
     "nationality": "International", "bats": "R",
     "bat_speed": 78.0, "max_exit_velo": 120.1, "pct90_exit_velo": 111.5,
     "pulled_barrel_pct": 0.102, "hr_per_pa": 0.070, "avg_exit_velo": 95.3,
     "total_barrel_rate": 0.13, "mean_launch_angle": 15.0, "total_balls_hit": 335},
]


def _manual_entry() -> list:
    """Interactive CLI to enter player stats without fetching Statcast."""
    print("\nManual player entry mode.  Enter stats for 8 players.\n")
    players = []
    for i in range(8):
        print(f"--- Player {i + 1} ---")
        name        = input("  Name: ").strip()
        team        = input("  Team (e.g. NYY): ").strip().upper()
        league      = input("  League (AL/NL): ").strip().upper()
        nationality = input("  Nationality (USA/International): ").strip()
        bats        = input("  Bats (R/L): ").strip().upper()
        bat_speed       = float(input("  Bat Speed (mph): "))
        max_exit_velo   = float(input("  Max Exit Velo (mph): "))
        pct90_exit_velo = float(input("  EV 90th Percentile (mph): "))
        total_balls_hit = int(input("  Total Balls Hit: "))
        pulled_fb_pct   = float(input("  Pulled FB% (0-1): "))
        players.append({
            "name": name, "mlb_id": None, "team": team,
            "league": league, "nationality": nationality, "bats": bats,
            "bat_speed": bat_speed, "max_exit_velo": max_exit_velo,
            "pct90_exit_velo": pct90_exit_velo,
            "total_balls_hit": total_balls_hit,
            "pulled_fb_pct": pulled_fb_pct,
        })
    return players


def _print_weighted_vars(players: list) -> None:
    """Print raw attribute values and normalized weighted contributions per player."""
    from config import ATTRIBUTE_WEIGHTS

    attr_keys = list(ATTRIBUTE_WEIGHTS.keys())
    weights   = np.array([ATTRIBUTE_WEIGHTS[k] for k in attr_keys])

    raw      = np.array([[p[k] for p in players] for k in attr_keys], dtype=float).T  # (n, attrs)
    col_min  = raw.min(axis=0)
    col_max  = raw.max(axis=0)
    denom    = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    normed   = (raw - col_min) / denom
    weighted = normed * weights  # (n, attrs)

    order = sorted(range(len(players)), key=lambda i: -players[i]["power_score"])

    short_labels = {
        "bat_speed":         "BatSpd",
        "max_exit_velo":     "MaxEV",
        "pulled_barrel_pct": "PullBrl%",
        "hr_per_pa":         "HR/PA",
        "pct90_exit_velo":   "EV90",
    }
    col_w = 10
    name_w = 23

    def _header(extra_col=False):
        h = f"  {'Player':<{name_w}}"
        for k in attr_keys:
            h += f"  {short_labels[k]:>{col_w}}"
        if extra_col:
            h += f"  {'Total':>{col_w}}"
        return h

    def _subheader(extra_col=False):
        s = f"  {'':>{name_w}}"
        for w in weights:
            s += f"  {f'({int(w*100)}%)':>{col_w}}"
        if extra_col:
            s += f"  {'':>{col_w}}"
        return s

    sep = "-" * (name_w + 4 + len(attr_keys) * (col_w + 2))

    # ── Raw attribute values ──────────────────────────────────
    print("\n=== Raw Attributes ===")
    print(_header())
    print(_subheader())
    print(sep)
    for i in order:
        p   = players[i]
        row = f"  {p['name']:<{name_w}}"
        for j, k in enumerate(attr_keys):
            v = raw[i, j]
            if k in ("pulled_barrel_pct", "hr_per_pa"):
                row += f"  {v:>{col_w}.3f}"
            else:
                row += f"  {v:>{col_w}.1f}"
        print(row)

    # ── Weighted contributions ────────────────────────────────
    sep2 = "-" * (name_w + 4 + (len(attr_keys) + 1) * (col_w + 2))
    print("\n=== Weighted Contributions (normalized × weight) ===")
    print(_header(extra_col=True))
    print(_subheader(extra_col=True))
    print(sep2)
    for i in order:
        p   = players[i]
        row = f"  {p['name']:<{name_w}}"
        for j in range(len(attr_keys)):
            row += f"  {weighted[i, j]:>{col_w}.4f}"
        row += f"  {weighted[i].sum():>{col_w}.4f}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="HRD 2026 Betting Model")
    parser.add_argument("--demo",   action="store_true", help="Run with placeholder demo players")
    parser.add_argument("--manual", action="store_true", help="Enter player stats manually")
    parser.add_argument("--sims",   type=int, default=N_SIMULATIONS,
                        help=f"Number of Monte Carlo simulations (default: {N_SIMULATIONS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Save market prices to this JSON file")
    args = parser.parse_args()

    # ── 1. Load players ────────────────────────────────────────
    if args.demo:
        print("\n[Demo mode] Using placeholder players — no Statcast fetch.\n")
        players = DEMO_PLAYERS

    elif args.manual:
        players = _manual_entry()

    else:
        if not PLAYERS:
            print(
                "\nNo players configured.  Either:\n"
                "  • Add players to PLAYERS in config.py\n"
                "  • Run with --demo for a test run\n"
                "  • Run with --manual to enter stats interactively\n"
            )
            sys.exit(1)

        print("\nFetching Statcast data from Baseball Savant...\n")
        players = load_player_attributes(PLAYERS)

    if len(players) != 8:
        print(f"ERROR: The HRD requires exactly 8 players (got {len(players)}).")
        sys.exit(1)

    # ── 2. Build player power models ───────────────────────────
    park = PARK_FACTORS.get(HRD_VENUE, PARK_FACTORS["Neutral"])
    print(f"\nVenue: {HRD_VENUE}  "
          f"(R factor: {park['hr_factor_R']:.2f}, "
          f"L factor: {park['hr_factor_L']:.2f}, "
          f"dist bonus: {park['dist_bonus_ft']:+.1f}ft)\n")
    print(f"Building player models...\n")
    players = build_player_models(players)

    print(f"{'Player':<25} {'Power Score':>12} {'HR Prob/Swing':>14} {'Mean Dist':>10}")
    print("-" * 65)
    for p in sorted(players, key=lambda x: -x["power_score"]):
        print(f"  {p['name']:<23} {p['power_score']:>12.4f} {p['hr_prob']:>13.1%} {p['mean_hr_dist']:>9.1f}ft")

    _print_weighted_vars(players)

    # ── 3. Simulate ────────────────────────────────────────────
    print(f"\nRunning {args.sims:,} simulations...")
    sims = run_simulations(players, n=args.sims)
    print("Done.\n")

    # ── 4. Price markets ───────────────────────────────────────
    print("Pricing markets...")
    markets = price_all_markets(players, sims)

    # ── 5. Output ──────────────────────────────────────────────
    print_markets(markets)

    if args.output:
        save_markets_json(markets, args.output)


if __name__ == "__main__":
    main()
