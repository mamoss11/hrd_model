# ─────────────────────────────────────────────────────────────
#  HRD Model — Player Power Rating
# ─────────────────────────────────────────────────────────────
import numpy as np

from config import (
    ATTRIBUTE_WEIGHTS,
    HR_PROB_MIN, HR_PROB_MAX,
    DIST_BASE, DIST_EV_BASE, DIST_EV_SCALE,
    DIST_BS_BASE, DIST_BS_SCALE, DIST_STD,
    EV_MEAN_FACTOR, EV_STD,
    HISTORICAL_DERBY, HISTORICAL_YEAR_WEIGHTS,
)


def build_player_models(players: list) -> list:
    """
    Given a list of player dicts (with Statcast attributes already loaded),
    return the same list augmented with:
      power_score      – raw weighted composite (before normalisation)
      hr_prob          – per-swing HR probability in the derby
      mean_hr_dist     – mean HR distance (feet)
      mean_hr_ev       – mean HR exit velocity in the derby
    """
    scores = _compute_power_scores(players)
    _attach_probabilities(players, scores)
    return players


# ── internal helpers ──────────────────────────────────────────

def _compute_power_scores(players: list) -> np.ndarray:
    """
    1. Normalise each attribute across the field (min–max → 0–1).
    2. Compute weighted sum.
    3. Apply historical derby adjustment where available.
    """
    attr_keys = list(ATTRIBUTE_WEIGHTS.keys())
    weights   = np.array([ATTRIBUTE_WEIGHTS[k] for k in attr_keys])

    raw = np.array([[p[k] for k in attr_keys] for p in players], dtype=float)

    # min-max normalise each attribute column
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1, col_max - col_min)
    normed  = (raw - col_min) / denom

    scores = normed @ weights  # shape: (n_players,)

    # Historical derby adjustment
    scores = _apply_historical_adjustment(players, scores)

    return scores


def _apply_historical_adjustment(players: list, scores: np.ndarray) -> np.ndarray:
    """
    Blend in historical derby HR-rate info.
    For each player with past derby data, compute a weighted historical
    HR-rate signal (0–1 normalised across field) and nudge the base score.
    """
    hist_signals = np.full(len(players), np.nan)

    for i, p in enumerate(players):
        hist = HISTORICAL_DERBY.get(p["mlb_id"])
        if not hist:
            continue
        years  = hist["years"]
        r1_hrs = hist["r1_hrs"]
        # weighted average of past round-1 HR totals
        total_w, total_whr = 0.0, 0.0
        for yr, hr in zip(years, r1_hrs):
            w = HISTORICAL_YEAR_WEIGHTS.get(yr, 0.1)
            total_w   += w
            total_whr += w * hr
        if total_w > 0:
            hist_signals[i] = total_whr / total_w

    valid = ~np.isnan(hist_signals)
    if valid.sum() >= 2:
        mn = hist_signals[valid].min()
        mx = hist_signals[valid].max()
        if mx > mn:
            hist_norm = np.where(valid, (hist_signals - mn) / (mx - mn), 0.5)
        else:
            hist_norm = np.full(len(players), 0.5)
        # Blend: 80% Statcast model, 20% historical
        scores = 0.80 * scores + 0.20 * np.where(valid, hist_norm, scores)

    return scores


def _attach_probabilities(players: list, scores: np.ndarray) -> None:
    """
    Map normalised power scores to per-swing HR probability and
    distance / EV distribution parameters.  Results written in-place.
    """
    s_min = scores.min()
    s_max = scores.max()
    denom = (s_max - s_min) if s_max > s_min else 1.0

    for i, p in enumerate(players):
        norm = (scores[i] - s_min) / denom   # 0→1

        # Per-swing HR probability (linear mapping)
        p["hr_prob"]     = HR_PROB_MIN + (HR_PROB_MAX - HR_PROB_MIN) * norm
        p["power_score"] = float(scores[i])

        # Distance model parameters
        ev = p["max_exit_velo"]
        bs = p["bat_speed"]
        mean_dist = (
            DIST_BASE
            + (ev - DIST_EV_BASE) * DIST_EV_SCALE
            + (bs - DIST_BS_BASE) * DIST_BS_SCALE
        )
        p["mean_hr_dist"] = float(mean_dist)
        p["dist_std"]     = DIST_STD

        # EV model parameters
        p["mean_hr_ev"]   = float(ev * EV_MEAN_FACTOR)
        p["ev_std"]       = EV_STD
