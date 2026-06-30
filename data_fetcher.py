# ─────────────────────────────────────────────────────────────
#  HRD Model — Statcast Data Fetcher
# ─────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
from pybaseball import statcast_batter, cache
cache.enable()

from config import SEASON_START, SEASON_END

warnings.filterwarnings("ignore")


def fetch_statcast(mlb_id: int, start: str = SEASON_START, end: str = SEASON_END) -> pd.DataFrame:
    """Return raw Statcast pitch-level data for one batter."""
    df = statcast_batter(start, end, player_id=mlb_id)
    return df


def compute_attributes(df: pd.DataFrame, bats: str) -> dict:
    """
    Derive the five model attributes from a raw Statcast DataFrame.

    Attributes
    ----------
    bat_speed          – mean bat speed on all swings with recorded bat speed
    max_exit_velo      – season-high exit velocity on any batted ball
    pulled_barrel_pct  – pulled barrels / total batted balls
    hr_per_pa          – home runs / plate appearances
    pct90_exit_velo    – 90th-percentile exit velocity on batted balls (mph)
    """
    # Batted balls only (type == 'X')
    batted = df[df["type"] == "X"].copy()

    if batted.empty:
        raise ValueError("No batted ball data found for this player/date range.")

    # ── bat speed ────────────────────────────────────────────
    if "bat_speed" in batted.columns:
        bs_vals = batted["bat_speed"].dropna()
        bat_speed = float(bs_vals.mean()) if not bs_vals.empty else np.nan
    else:
        bat_speed = np.nan

    # ── exit velocity ────────────────────────────────────────
    ev = batted["launch_speed"].dropna()
    max_exit_velo   = float(ev.max())              if not ev.empty else np.nan
    pct90_exit_velo = float(ev.quantile(0.90))     if not ev.empty else np.nan

    # ── hard-hit rate (>=110 mph) — kept for reference ────────
    total_balls_hit   = int(ev.count())
    balls_110_plus    = int((ev >= 110).sum())
    hard_hit_rate_110 = balls_110_plus / total_balls_hit if total_balls_hit > 0 else 0.0

    # ── HR per PA ─────────────────────────────────────────────
    # A PA ends whenever the 'events' column is non-null
    total_pa = int(df["events"].notna().sum())
    hr_count = int((df["events"] == "home_run").sum())
    hr_per_pa = hr_count / total_pa if total_pa > 0 else 0.0

    # ── avg exit velocity ─────────────────────────────────────
    avg_exit_velo = float(ev.mean()) if not ev.empty else np.nan

    # ── pulled barrel % ───────────────────────────────────────
    # Barrels: launch_speed_angle == 6 (Statcast barrel classification)
    # Pulled:  RHB hits to left field (hc_x < 125), LHB to right (hc_x > 125)
    if "launch_speed_angle" in batted.columns:
        barrels = batted[batted["launch_speed_angle"] == 6].dropna(subset=["hc_x"])
    else:
        barrels = pd.DataFrame()

    if barrels.empty or total_balls_hit == 0:
        pulled_barrel_pct = 0.0
        total_barrel_rate = 0.0
    else:
        if bats == "R":
            pulled_barrels = (barrels["hc_x"] < 125).sum()
        else:
            pulled_barrels = (barrels["hc_x"] > 125).sum()
        pulled_barrel_pct = float(pulled_barrels / total_balls_hit)
        total_barrel_rate = float(len(barrels) / total_balls_hit)

    # ── mean launch angle + ideal LA % (25–35°) ──────────────
    if "launch_angle" in batted.columns:
        la_vals = batted["launch_angle"].dropna()
        mean_launch_angle = float(la_vals.mean()) if not la_vals.empty else np.nan
        ideal_la_pct = float(((la_vals >= 25) & (la_vals <= 35)).sum() / len(la_vals)) if not la_vals.empty else 0.0
    else:
        mean_launch_angle = np.nan
        ideal_la_pct = 0.0

    return {
        "bat_speed":          bat_speed,
        "max_exit_velo":      max_exit_velo,
        "pct90_exit_velo":    pct90_exit_velo,
        "avg_exit_velo":      avg_exit_velo,
        "pulled_barrel_pct":  pulled_barrel_pct,
        "total_barrel_rate":  total_barrel_rate,
        "hr_per_pa":          hr_per_pa,
        "hard_hit_rate_110":  hard_hit_rate_110,
        "mean_launch_angle":  mean_launch_angle,
        "ideal_la_pct":       ideal_la_pct,
        "total_balls_hit":    total_balls_hit,
    }


def load_player_attributes(players: list, start: str = SEASON_START, end: str = SEASON_END) -> list:
    """
    Fetch and compute Statcast attributes for every player in the field.
    Returns a list of player dicts with attributes merged in.
    """
    enriched = []
    for p in players:
        p_start = p.get("season_start", start)
        p_end   = p.get("season_end",   end)
        print(f"  Fetching Statcast data for {p['name']} (ID {p['mlb_id']})...")
        if p_start != start or p_end != end:
            print(f"    (custom window: {p_start} -> {p_end})")
        try:
            df    = fetch_statcast(p["mlb_id"], p_start, p_end)
            attrs = compute_attributes(df, p["bats"])
        except Exception as exc:
            print(f"    WARNING: No Statcast data for {p['name']} (2025–2026): {exc}")
            attrs = {
                "bat_speed":          np.nan,
                "max_exit_velo":      np.nan,
                "pct90_exit_velo":    np.nan,
                "avg_exit_velo":      np.nan,
                "pulled_barrel_pct":  np.nan,
                "total_barrel_rate":  np.nan,
                "hr_per_pa":          np.nan,
                "hard_hit_rate_110":  np.nan,
                "mean_launch_angle":  np.nan,
                "ideal_la_pct":       0.0,
                "total_balls_hit":    0,
            }
        enriched.append({**p, **attrs})

    _fill_missing_with_field_mean(enriched)
    return enriched


# ── helpers ───────────────────────────────────────────────────

def _fill_missing_with_field_mean(players: list) -> None:
    """Replace NaN attribute values with the field mean for that attribute."""
    attr_keys = ["bat_speed", "max_exit_velo", "pct90_exit_velo", "avg_exit_velo",
                 "pulled_barrel_pct", "total_barrel_rate", "hr_per_pa",
                 "hard_hit_rate_110", "mean_launch_angle", "ideal_la_pct"]
    for key in attr_keys:
        vals = [p[key] for p in players if not np.isnan(p[key])]
        if not vals:
            # Entire field missing this attribute (e.g. bat_speed pre-2023)
            # Set to 0 so min-max normalisation treats it as a non-discriminator
            for p in players:
                p[key] = 0.0
            continue
        mean_val = float(np.mean(vals))
        for p in players:
            if np.isnan(p[key]):
                print(f"    INFO: Using field mean ({mean_val:.2f}) for {p['name']} – {key}")
                p[key] = mean_val
