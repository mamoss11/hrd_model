# ─────────────────────────────────────────────────────────────
#  HRD Betting Model — Streamlit App
# ─────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="HRD Betting Model",
    page_icon="baseball",
    layout="wide",
)

import config as cfg
import simulator as _sim_mod
import player_model as _pm_mod
from data_fetcher import load_player_attributes
from player_model import build_player_models
from markets import price_all_markets
import requests as _requests
from mlb_lookup import search_players, get_player_info


def load_roster(season: int = 2026) -> list:
    resp = _requests.get(
        "https://statsapi.mlb.com/api/v1/sports/1/players",
        params={"season": season, "gameType": "R"},
        timeout=20,
    )
    resp.raise_for_status()
    people = resp.json().get("people", [])
    return sorted(
        [{"id": p["id"], "fullName": p["fullName"]} for p in people],
        key=lambda x: x["fullName"],
    )


# ── Demo players (all required attributes pre-filled) ──────────────────
_DEMO_PLAYERS = [
    {"name": "Player A", "mlb_id": None, "team": "NYY", "league": "AL",
     "nationality": "USA", "bats": "R",
     "bat_speed": 77.5, "max_exit_velo": 119.4, "pct90_exit_velo": 109.5,
     "pulled_barrel_pct": 0.095, "hr_per_pa": 0.065, "avg_exit_velo": 94.1,
     "total_barrel_rate": 0.12, "mean_launch_angle": 14.2, "total_balls_hit": 320},
    {"name": "Player B", "mlb_id": None, "team": "LAD", "league": "NL",
     "nationality": "International", "bats": "R",
     "bat_speed": 76.1, "max_exit_velo": 117.8, "pct90_exit_velo": 108.0,
     "pulled_barrel_pct": 0.082, "hr_per_pa": 0.058, "avg_exit_velo": 92.8,
     "total_barrel_rate": 0.10, "mean_launch_angle": 13.5, "total_balls_hit": 290},
    {"name": "Player C", "mlb_id": None, "team": "ATL", "league": "NL",
     "nationality": "USA", "bats": "R",
     "bat_speed": 74.8, "max_exit_velo": 116.2, "pct90_exit_velo": 106.5,
     "pulled_barrel_pct": 0.078, "hr_per_pa": 0.052, "avg_exit_velo": 91.5,
     "total_barrel_rate": 0.09, "mean_launch_angle": 12.8, "total_balls_hit": 310},
    {"name": "Player D", "mlb_id": None, "team": "HOU", "league": "AL",
     "nationality": "USA", "bats": "R",
     "bat_speed": 73.5, "max_exit_velo": 115.0, "pct90_exit_velo": 105.0,
     "pulled_barrel_pct": 0.071, "hr_per_pa": 0.048, "avg_exit_velo": 90.2,
     "total_barrel_rate": 0.08, "mean_launch_angle": 12.1, "total_balls_hit": 280},
    {"name": "Player E", "mlb_id": None, "team": "NYM", "league": "NL",
     "nationality": "International", "bats": "L",
     "bat_speed": 75.2, "max_exit_velo": 116.9, "pct90_exit_velo": 107.0,
     "pulled_barrel_pct": 0.086, "hr_per_pa": 0.055, "avg_exit_velo": 92.0,
     "total_barrel_rate": 0.11, "mean_launch_angle": 13.9, "total_balls_hit": 300},
    {"name": "Player F", "mlb_id": None, "team": "MIL", "league": "NL",
     "nationality": "USA", "bats": "R",
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


# ── Market categorisation ────────────────────────────────────────────────
_TOURNAMENT = {
    "Winner", "Make the Final", "Make the Semi Finals",
    "Name the Finalists", "Final Exact Result", "Semifinal Matchups",
}
_TOTALS = {
    "Total HRs", "Total HRs by Winning Player", "Total HRs in Round 1",
    "Total Swing-offs", "Most HRs by Any Player in Round 1",
    "Player to Hit Most HRs in Round 1",
}
_DIST_FIXED = {
    "Player to Hit Longest HR", "Length of Longest HR",
    "Will the Distance Record be Broken?",
}
_EV_FIXED = {
    "Player with Highest Exit Velocity HR", "Highest Exit Velocity HR",
}
_CATEGORY = {
    "League of Winner", "League with Most Derby HRs",
    "AL Total HRs", "NL Total HRs",
    "Winner - USA v Rest of the World", "Winner - Batter Handedness",
}
_GROUP_ORDER = [
    "Tournament", "Totals", "Player Props",
    "Head-to-Head", "Distance", "Exit Velocity", "Category",
]


def _categorize(name, player_names):
    if name in _TOURNAMENT:   return "Tournament"
    if name in _TOTALS:       return "Totals"
    if name in _DIST_FIXED:   return "Distance"
    if name in _EV_FIXED:     return "Exit Velocity"
    if name in _CATEGORY:     return "Category"
    if "Round 1 Most HRs -" in name:   return "Head-to-Head"
    if "Round 1 Longest HR -" in name: return "Head-to-Head"
    for nm in player_names:
        if name.startswith(nm):
            if "Distance" in name or "Longest" in name: return "Distance"
            if "Exit Velocity" in name:                  return "Exit Velocity"
            return "Player Props"
    return "Totals"


def _group_markets(markets, player_names):
    groups = {k: {} for k in _GROUP_ORDER}
    for name, outcomes in markets.items():
        groups[_categorize(name, player_names)][name] = outcomes
    return {k: v for k, v in groups.items() if v}


def _outcomes_df(outcomes):
    rows = []
    for label, data in sorted(outcomes.items(), key=lambda x: -x[1]["prob"]):
        rows.append({
            "Outcome":  str(label),
            "Prob %":   f"{data['prob'] * 100:.2f}%",
            "American": data["american"],
        })
    return pd.DataFrame(rows)


def _render_group(group_dict, container):
    with container:
        for market_name, outcomes in sorted(group_dict.items()):
            with st.expander(market_name):
                st.dataframe(
                    _outcomes_df(outcomes),
                    width="stretch",
                    hide_index=True,
                )


import re as _re
_THRESH_RE = _re.compile(r"^(.+) (\d+)\+ HRs in Round 1$")


def _render_player_props(group_dict, player_names, container):
    threshold_markets = {}
    other_markets = {}
    for mkt_name, outcomes in group_dict.items():
        m = _THRESH_RE.match(mkt_name)
        if m:
            threshold_markets[(m.group(1), int(m.group(2)))] = outcomes
        else:
            other_markets[mkt_name] = outcomes

    with container:
        if threshold_markets:
            st.markdown("**R1 HR Thresholds**")
            thresholds = sorted({t for _, t in threshold_markets})
            rows = []
            for thresh in thresholds:
                line = thresh - 0.5
                row = {"Threshold": f"Over {line}"}
                for nm in player_names:
                    d = threshold_markets.get((nm, thresh), {}).get(f"Over {line}")
                    row[nm] = f"{d['prob']*100:.1f}% ({d['american']})" if d else "—"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if other_markets:
                st.divider()

        for mkt_name, outcomes in sorted(other_markets.items()):
            with st.expander(mkt_name):
                st.dataframe(_outcomes_df(outcomes), width="stretch", hide_index=True)


# ── Sidebar ──────────────────────────────────────────────────────────────
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "player_attrs.json")
venues = list(cfg.PARK_FACTORS.keys())

with st.sidebar:
    st.title("Configuration")

    st.subheader("Venue")
    venue = st.selectbox(
        "Venue",
        venues,
        index=venues.index(cfg.HRD_VENUE),
        label_visibility="collapsed",
    )
    park = cfg.PARK_FACTORS[venue]
    st.caption(
        f"R factor: {park['hr_factor_R']:.2f} | "
        f"L factor: {park['hr_factor_L']:.2f} | "
        f"Dist bonus: {park['dist_bonus_ft']:+.1f} ft"
    )

    st.divider()
    st.subheader("Simulation")
    n_sims = st.select_slider(
        "Simulations",
        options=[10_000, 25_000, 50_000, 100_000],
        value=100_000,
        format_func=lambda x: f"{x:,}",
    )
    perf_std = st.slider(
        "Performance Std Dev", 0.10, 0.50,
        float(cfg.PERFORMANCE_STD), 0.01,
        help="Higher = more upsets, flatter prices",
    )

    st.divider()
    st.subheader("HR Probability Range")
    hr_min = st.slider("Min (weakest player)", 0.25, 0.55, float(cfg.HR_PROB_MIN), 0.01)
    hr_max = st.slider("Max (strongest player)", 0.40, 0.75, float(cfg.HR_PROB_MAX), 0.01)
    if hr_min >= hr_max:
        st.error("Min must be less than Max.")

    st.divider()
    st.subheader("Attribute Weights")
    w_bat  = st.slider("Bat Speed",        0.0, 0.60, float(cfg.ATTRIBUTE_WEIGHTS["bat_speed"]),          0.01)
    w_ev   = st.slider("Max Exit Velocity", 0.0, 0.60, float(cfg.ATTRIBUTE_WEIGHTS["max_exit_velo"]),      0.01)
    w_pull = st.slider("Pulled Barrel %",  0.0, 0.60, float(cfg.ATTRIBUTE_WEIGHTS["pulled_barrel_pct"]),  0.01)
    w_hr   = st.slider("HR / PA",          0.0, 0.60, float(cfg.ATTRIBUTE_WEIGHTS["hr_per_pa"]),           0.01)
    w_hh   = st.slider("EV 90th Pctl",      0.0, 0.60, float(cfg.ATTRIBUTE_WEIGHTS["pct90_exit_velo"]),   0.01)
    raw_sum = w_bat + w_ev + w_pull + w_hr + w_hh
    if raw_sum > 0 and abs(raw_sum - 1.0) > 0.005:
        st.info(f"Weights sum to {raw_sum:.2f} — auto-normalized on run.")

    st.divider()
    st.subheader("Data Source")
    _cache_available = os.path.exists(_CACHE_PATH)
    _data_opts = (
        ["Load from cache", "Fetch Statcast", "Demo mode"]
        if _cache_available else
        ["Fetch Statcast", "Demo mode"]
    )
    data_mode = st.radio("Player data", _data_opts)
    if data_mode == "Fetch Statcast":
        season_start = st.text_input("Season start", "2025-03-27", key="season_start_v3")
        season_end   = st.text_input("Season end",   "2026-06-25", key="season_end_v3")
    else:
        season_start = season_end = None
    if data_mode == "Load from cache":
        st.caption("Using pre-built player cache. Run `build_cache.py` to refresh.")


# ── Main ─────────────────────────────────────────────────────────────────
st.title("HRD Betting Model")

# ── Cached MLB roster + player lookup ────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="Loading MLB roster...")
def _load_roster():
    return load_roster()

def _fetch_player(mlb_id: int):
    try:
        result = get_player_info(mlb_id)
        return result
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_attrs(players_cfg, start, end):
    return load_player_attributes([dict(p) for p in players_cfg], start, end)

# ── Player Field ──────────────────────────────────────────────────────────
st.subheader("Player Field")

try:
    roster = _load_roster()
except Exception as e:
    roster = []
    st.error(f"Could not load MLB roster: {e}")

_BLANK       = "— select player —"
_OPTIONS     = [_BLANK] + [p["fullName"] for p in roster]
_ID_MAP      = {p["fullName"]: p["id"] for p in roster}
_CFG_BY_NAME = {p["name"]: p for p in cfg.PLAYERS}
_CFG_BY_ID   = {p["mlb_id"]: p for p in cfg.PLAYERS if p.get("mlb_id")}

# Header row
hcols = st.columns([4, 1, 1, 1, 2])
for col, lbl in zip(hcols, ["Player", "Team", "Lg", "Bats", "Nationality"]):
    col.markdown(f"**{lbl}**")

players_for_run = []

for i in range(8):
    cols = st.columns([4, 1, 1, 1, 2])

    cfg_name    = cfg.PLAYERS[i]["name"] if i < len(cfg.PLAYERS) else _BLANK
    default_idx = _OPTIONS.index(cfg_name) if cfg_name in _OPTIONS else 0

    with cols[0]:
        chosen = st.selectbox(
            f"Player {i + 1}",
            _OPTIONS,
            index=default_idx,
            label_visibility="collapsed",
            key=f"slot_{i}",
        )

    if chosen == _BLANK:
        continue

    # Use config data if available — match by name first, then by MLB ID
    roster_id = _ID_MAP.get(chosen)
    cfg_p = _CFG_BY_NAME.get(chosen) or _CFG_BY_ID.get(roster_id)
    if cfg_p:
        name, mlb_id    = cfg_p["name"], cfg_p.get("mlb_id")
        team, league    = cfg_p["team"], cfg_p["league"]
        bats, nat       = cfg_p["bats"], cfg_p["nationality"]
        is_switch       = False
    else:
        info   = _fetch_player(roster_id) if roster_id else None
        if not info:
            cols[1].markdown("—")
            continue
        if "_error" in info:
            cols[1].markdown(f"⚠ {info['_error'][:60]}")
            continue
        name, mlb_id    = info["name"], info["mlb_id"]
        team, league    = info["team"], info["league"]
        bats, nat       = info["bats"], info["nationality"]
        is_switch       = info.get("is_switch", False)
        if league not in ("AL", "NL"):
            league = "AL"

    cols[1].markdown(f"**{team}**")
    cols[2].markdown(f"**{league}**")
    cols[3].markdown(f"**{bats}**" + (" *(sw)*" if is_switch else ""))
    cols[4].markdown(nat)

    players_for_run.append({
        "name":        name,
        "mlb_id":      mlb_id,
        "team":        team,
        "league":      league,
        "nationality": nat,
        "bats":        bats,
    })

n_players = len(players_for_run)
if n_players != 8:
    st.warning(f"Exactly 8 players required — currently {n_players}.")

# ── Run button ────────────────────────────────────────────────────────────
run_ok = (n_players == 8 or data_mode == "Demo mode") and (hr_min < hr_max) and (raw_sum > 0)

if st.button("Run Simulation", type="primary", disabled=not run_ok):

    players_cfg = players_for_run

    # Normalise weights
    denom   = raw_sum
    weights = {
        "bat_speed":         w_bat  / denom,
        "max_exit_velo":     w_ev   / denom,
        "pulled_barrel_pct": w_pull / denom,
        "hr_per_pa":         w_hr   / denom,
        "pct90_exit_velo":   w_hh   / denom,
    }

    # Patch module globals before running
    _sim_mod._PARK_HR_FACTOR_R = park["hr_factor_R"]
    _sim_mod._PARK_HR_FACTOR_L = park["hr_factor_L"]
    _sim_mod._PARK_DIST_BONUS  = park["dist_bonus_ft"]
    _sim_mod.PERFORMANCE_STD   = perf_std
    _pm_mod.HR_PROB_MIN        = hr_min
    _pm_mod.HR_PROB_MAX        = hr_max
    _pm_mod.ATTRIBUTE_WEIGHTS  = weights

    try:
        # 1. Load player data
        if data_mode == "Demo mode":
            players = _DEMO_PLAYERS
        elif data_mode == "Load from cache":
            with open(_CACHE_PATH) as _f:
                _all_cached = json.load(_f)
            _field_ids   = {p["mlb_id"] for p in players_cfg if p.get("mlb_id")}
            _field_names = {p["name"]   for p in players_cfg}
            players = [p for p in _all_cached
                       if p.get("mlb_id") in _field_ids or p["name"] in _field_names]
            if len(players) != len(players_cfg):
                st.error(
                    f"Cache matched {len(players)} of {len(players_cfg)} players. "
                    "The cache was built for a different field. "
                    "Switch to **Fetch Statcast** mode, or rebuild the cache locally with `build_cache.py`."
                )
                st.stop()
        else:
            with st.spinner("Fetching Statcast data from Baseball Savant..."):
                players = _cached_load_attrs(
                    tuple(tuple(sorted(p.items())) for p in players_cfg),
                    season_start, season_end,
                )

        # 2. Build player power models
        with st.spinner("Building player models..."):
            players = build_player_models(players)

        # 3. Run Monte Carlo simulations
        with st.spinner(f"Running {n_sims:,} simulations..."):
            sims = _sim_mod.run_simulations(players, n=n_sims)

        # 4. Price markets
        with st.spinner("Pricing markets..."):
            markets = price_all_markets(players, sims)

        st.session_state["results"] = {
            "players": players,
            "markets": markets,
            "venue":   venue,
            "n_sims":  n_sims,
        }

    except Exception as exc:
        st.error(f"Model error: {exc}")
        raise


# ── Results ───────────────────────────────────────────────────────────────
if "results" in st.session_state:
    res     = st.session_state["results"]
    players = res["players"]
    markets = res["markets"]
    names   = [p["name"] for p in players]

    st.divider()
    st.subheader("Player Ratings")
    st.caption(
        f"Venue: **{res['venue']}**  |  "
        f"Simulations: **{res['n_sims']:,}**  |  "
        f"HR Prob range: **{hr_min:.0%} – {hr_max:.0%}**"
    )

    ratings_df = pd.DataFrame([
        {
            "Player":          p["name"],
            "Team":            p["team"],
            "League":          p["league"],
            "Bats":            p["bats"],
            "Nationality":     p["nationality"],
            "Power Score":     round(p["power_score"], 4),
            "HR Prob/Swing":   f"{p['hr_prob']:.1%}",
            "Mean HR Dist ft": f"{p['mean_hr_dist']:.1f}",
        }
        for p in sorted(players, key=lambda x: -x["power_score"])
    ])
    st.dataframe(ratings_df, width="stretch", hide_index=True)

    # ── Market tabs ───────────────────────────────────────────────────────
    st.subheader("Markets")
    grouped   = _group_markets(markets, names)
    tab_names = list(grouped.keys())
    tabs      = st.tabs(tab_names)

    for tab, group_name in zip(tabs, tab_names):
        if group_name == "Player Props":
            _render_player_props(grouped[group_name], names, tab)
        else:
            _render_group(grouped[group_name], tab)
