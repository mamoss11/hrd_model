# ─────────────────────────────────────────────────────────────
#  Quick script: print Winner / Make Final / Make Semis for all
#  four backtest years using cached Statcast data.
# ─────────────────────────────────────────────────────────────
import socket, sys
socket.setdefaulttimeout(120)

import simulator as sim_module
from data_fetcher import load_player_attributes
from player_model import build_player_models
from simulator import run_simulations
from markets import price_all_markets

YEARS = {
    2022: {
        "venue": "Nationals Park",
        "park": {"hr_factor_R": 1.00, "hr_factor_L": 1.02, "dist_bonus_ft": 0.0},
        "season_start": "2022-04-07", "season_end": "2022-10-05",
        "winner": "Juan Soto",
        "players": [
            {"name": "Kyle Schwarber",   "mlb_id": 656941, "team": "PHI", "league": "NL", "nationality": "USA",           "bats": "L"},
            {"name": "Albert Pujols",    "mlb_id": 405395, "team": "STL", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Juan Soto",        "mlb_id": 665742, "team": "WSH", "league": "NL", "nationality": "International", "bats": "L"},
            {"name": "Jose Ramirez",     "mlb_id": 608070, "team": "CLE", "league": "AL", "nationality": "International", "bats": "S"},
            {"name": "Corey Seager",     "mlb_id": 608369, "team": "TEX", "league": "AL", "nationality": "USA",           "bats": "L"},
            {"name": "Julio Rodriguez",  "mlb_id": 677594, "team": "SEA", "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Pete Alonso",      "mlb_id": 624413, "team": "NYM", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Ronald Acuna Jr.", "mlb_id": 660670, "team": "ATL", "league": "NL", "nationality": "International", "bats": "R"},
        ],
    },
    2023: {
        "venue": "T-Mobile Park",
        "park": {"hr_factor_R": 0.96, "hr_factor_L": 0.90, "dist_bonus_ft": 0.0},
        "season_start": "2023-03-30", "season_end": "2023-10-01",
        "winner": "Vlad Guerrero Jr.",
        "players": [
            {"name": "Luis Robert Jr.",   "mlb_id": 673357, "team": "CWS", "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Adley Rutschman",   "mlb_id": 668939, "team": "BAL", "league": "AL", "nationality": "USA",           "bats": "S"},
            {"name": "Adolis Garcia",     "mlb_id": 666969, "team": "TEX", "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Randy Arozarena",   "mlb_id": 668227, "team": "TB",  "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Mookie Betts",      "mlb_id": 605141, "team": "LAD", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Vlad Guerrero Jr.", "mlb_id": 665489, "team": "TOR", "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Pete Alonso",       "mlb_id": 624413, "team": "NYM", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Julio Rodriguez",   "mlb_id": 677594, "team": "SEA", "league": "AL", "nationality": "International", "bats": "R"},
        ],
    },
    2024: {
        "venue": "Globe Life Field",
        "park": {"hr_factor_R": 1.04, "hr_factor_L": 1.05, "dist_bonus_ft": 0.0},
        "season_start": "2024-03-20", "season_end": "2024-09-29",
        "winner": "Teoscar Hernandez",
        "players": [
            {"name": "Teoscar Hernandez", "mlb_id": 606192, "team": "LAD", "league": "NL", "nationality": "International", "bats": "R"},
            {"name": "Bobby Witt Jr.",    "mlb_id": 677951, "team": "KC",  "league": "AL", "nationality": "USA",           "bats": "R"},
            {"name": "Alec Bohm",         "mlb_id": 664761, "team": "PHI", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Jose Ramirez",      "mlb_id": 608070, "team": "CLE", "league": "AL", "nationality": "International", "bats": "S"},
            {"name": "Adolis Garcia",     "mlb_id": 666969, "team": "TEX", "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Marcell Ozuna",     "mlb_id": 542303, "team": "ATL", "league": "NL", "nationality": "International", "bats": "R"},
            {"name": "Pete Alonso",       "mlb_id": 624413, "team": "NYM", "league": "NL", "nationality": "USA",           "bats": "R"},
            {"name": "Gunnar Henderson",  "mlb_id": 683002, "team": "BAL", "league": "AL", "nationality": "USA",           "bats": "L"},
        ],
    },
    2025: {
        "venue": "Truist Park",
        "park": {"hr_factor_R": 1.06, "hr_factor_L": 1.07, "dist_bonus_ft": 0.0},
        "season_start": "2025-03-27", "season_end": "2025-10-01",
        "winner": "Cal Raleigh",
        "players": [
            {"name": "Cal Raleigh",       "mlb_id": 663728, "team": "SEA", "league": "AL", "nationality": "USA",           "bats": "L"},
            {"name": "Oneil Cruz",        "mlb_id": 665833, "team": "PIT", "league": "NL", "nationality": "International", "bats": "L"},
            {"name": "Junior Caminero",   "mlb_id": 691406, "team": "TB",  "league": "AL", "nationality": "International", "bats": "R"},
            {"name": "Byron Buxton",      "mlb_id": 621439, "team": "MIN", "league": "AL", "nationality": "USA",           "bats": "R"},
            {"name": "James Wood",        "mlb_id": 695578, "team": "WSH", "league": "NL", "nationality": "USA",           "bats": "L"},
            {"name": "Brent Rooker",      "mlb_id": 667670, "team": "OAK", "league": "AL", "nationality": "USA",           "bats": "R"},
            {"name": "Jazz Chisholm Jr.", "mlb_id": 665862, "team": "NYY", "league": "AL", "nationality": "USA",           "bats": "L"},
            {"name": "Matt Olson",        "mlb_id": 621566, "team": "ATL", "league": "NL", "nationality": "USA",           "bats": "L"},
        ],
    },
}

KEY_MARKETS = ["Winner", "Make the Final", "Make the Semi Finals"]


def run_year(year, cfg):
    players = load_player_attributes(cfg["players"], start=cfg["season_start"], end=cfg["season_end"])
    players = build_player_models(players)
    sim_module._PARK_HR_FACTOR_R = cfg["park"]["hr_factor_R"]
    sim_module._PARK_HR_FACTOR_L = cfg["park"]["hr_factor_L"]
    sim_module._PARK_DIST_BONUS  = cfg["park"]["dist_bonus_ft"]
    sims    = run_simulations(players, n=100000)
    markets = price_all_markets(players, sims)

    print(f"\n{'='*70}", flush=True)
    print(f"  {year} HRD — {cfg['venue']}   (actual winner: {cfg['winner']})", flush=True)
    print(f"{'='*70}", flush=True)

    for mkt_name in KEY_MARKETS:
        mkt    = markets.get(mkt_name, {})
        ranked = sorted(mkt.items(), key=lambda x: x[1]["prob"], reverse=True)
        print(f"\n  {mkt_name}", flush=True)
        print(f"  {'-'*54}", flush=True)
        for name, data in ranked:
            marker = " <-- WINNER" if name == cfg["winner"] else ""
            prob   = data["prob"] * 100
            odds   = data["american"]
            print(f"  {str(name):<28} {prob:5.1f}%  {odds:>9}{marker}", flush=True)


if __name__ == "__main__":
    for year in sorted(YEARS):
        run_year(year, YEARS[year])
    print(f"\n{'='*70}\n", flush=True)
