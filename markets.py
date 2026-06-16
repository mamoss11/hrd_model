# ─────────────────────────────────────────────────────────────
#  HRD Model — Market Pricing
# ─────────────────────────────────────────────────────────────
"""
All markets are priced as true implied probabilities (no vig).
American odds are derived from those probabilities.
"""
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict


def price_all_markets(players: list, sims: list) -> dict:
    """
    Master function.  Returns a nested dict:
      { market_name: { outcome_label: {"prob": float, "american": str} } }
    """
    n      = len(sims)
    names  = [p["name"] for p in players]
    p_info = {p["name"]: p for p in players}

    markets = {}

    markets.update(_market_winner(names, sims, n))
    markets.update(_market_make_final(names, sims, n))
    markets.update(_market_name_finalists(names, sims, n))
    markets.update(_market_final_exact_result(names, sims, n))
    markets.update(_market_make_semis(names, sims, n))
    markets.update(_market_semifinal_matchups(names, sims, n))
    markets.update(_market_total_hrs(sims, n))
    markets.update(_market_total_hrs_winning_player(sims, n))
    markets.update(_market_total_swingoffs(sims, n))
    markets.update(_market_total_r1_hrs(sims, n))
    markets.update(_market_most_r1_hrs_player(names, sims, n))
    markets.update(_market_most_r1_hrs_any(sims, n))
    markets.update(_market_player_r1_hrs(names, sims, n))
    markets.update(_market_player_r1_threshold(names, sims, n, threshold=15))
    markets.update(_market_player_r1_threshold(names, sims, n, threshold=20))
    markets.update(_market_player_r1_threshold(names, sims, n, threshold=25))
    markets.update(_market_player_r1_threshold(names, sims, n, threshold=30))
    markets.update(_market_r1_h2h(names, sims, n))
    markets.update(_market_r1_longest_h2h(names, sims, n))
    markets.update(_market_longest_hr_player(names, sims, n))
    markets.update(_market_longest_hr_distance(sims, n))
    markets.update(_market_player_longest_distance(names, sims, n))
    markets.update(_market_record_broken(sims, n))
    markets.update(_market_highest_ev_player(names, sims, n))
    markets.update(_market_highest_ev_value(sims, n))
    markets.update(_market_player_highest_ev(names, sims, n))
    markets.update(_market_winner_league(sims, n))
    markets.update(_market_league_most_hrs(sims, n))
    markets.update(_market_league_total_hrs(sims, n))
    markets.update(_market_winner_nationality(sims, n))
    markets.update(_market_winner_handedness(sims, n))

    return markets


# ── tournament markets ─────────────────────────────────────────

def _market_winner(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["winner"]] += 1
    return {"Winner": _probs_to_market(names, counts, n)}


def _market_make_final(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        for f in s["finalists"]:
            counts[f] += 1
    return {"Make the Final": _probs_to_market(names, counts, n)}


def _market_name_finalists(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["finalists"]] += 1   # tuple of 2 sorted names
    pairs = [tuple(sorted(c)) for c in combinations(names, 2)]
    return {"Name the Finalists": _probs_to_market(pairs, counts, n)}


def _market_final_exact_result(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        winner = s["winner"]
        loser  = [f for f in s["finalists"] if f != winner][0]
        counts[(winner, loser)] += 1
    perms = list(permutations(names, 2))
    return {"Final Exact Result": _probs_to_market(perms, counts, n)}


def _market_make_semis(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        for nm in s["semifinalists"]:
            counts[nm] += 1
    return {"Make the Semi Finals": _probs_to_market(names, counts, n)}


def _market_semifinal_matchups(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        # semi_matchups is a tuple of two sorted pairs
        key = tuple(sorted(s["semi_matchups"]))
        counts[key] += 1
    all_keys = list(counts.keys())  # enumerate observed matchups
    return {"Semifinal Matchups": _probs_to_market(all_keys, counts, n)}


# ── total HR markets ───────────────────────────────────────────

def _market_total_hrs(sims, n):
    vals = [s["total_derby_hrs"] for s in sims]
    return {"Total HRs": _over_under_lines(vals)}


def _market_total_hrs_winning_player(sims, n):
    vals = [s["winner_total_hrs"] for s in sims]
    return {"Total HRs by Winning Player": _over_under_lines(vals)}


def _market_total_swingoffs(sims, n):
    vals = [s["swing_offs"] for s in sims]
    return {"Total Swing-offs": _over_under_lines(vals)}


def _market_total_r1_hrs(sims, n):
    vals = [s["total_r1_hrs"] for s in sims]
    return {"Total HRs in Round 1": _over_under_lines(vals)}


# ── player round 1 markets ─────────────────────────────────────

def _market_most_r1_hrs_player(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["most_r1_hrs_player"]] += 1
    return {"Player to Hit Most HRs in Round 1": _probs_to_market(names, counts, n)}


def _market_most_r1_hrs_any(sims, n):
    vals = [s["most_r1_hrs_count"] for s in sims]
    return {"Most HRs by Any Player in Round 1": _over_under_lines(vals)}


def _market_player_r1_hrs(names, sims, n):
    market = {}
    for nm in names:
        vals = [s["r1_hrs"][nm] for s in sims]
        market[f"{nm} Total HRs in Round 1"] = _over_under_lines(vals)
    return market


def _market_player_r1_threshold(names, sims, n, threshold):
    market = {}
    for nm in names:
        count = sum(1 for s in sims if s["r1_hrs"][nm] >= threshold)
        prob  = count / n
        market[f"{nm} {threshold}+ HRs in Round 1"] = {
            "Yes": _fmt(prob),
            "No":  _fmt(1 - prob),
        }
    return market


# ── head-to-head round 1 ──────────────────────────────────────

def _market_r1_h2h(names, sims, n):
    market = {}
    for a, b in combinations(names, 2):
        counts = defaultdict(int)
        for s in sims:
            ha, hb = s["r1_hrs"][a], s["r1_hrs"][b]
            if ha > hb:
                counts[a] += 1
            elif hb > ha:
                counts[b] += 1
            else:
                counts["Tie"] += 1
        outcomes = [a, b, "Tie"]
        key = f"Round 1 Most HRs - {a} v {b}"
        market[key] = _probs_to_market(outcomes, counts, n)
    return market


# ── distance markets ───────────────────────────────────────────

def _market_r1_longest_h2h(names, sims, n):
    market = {}
    for a, b in combinations(names, 2):
        counts = defaultdict(int)
        for s in sims:
            da = s["r1_longest"].get(a, 0)
            db = s["r1_longest"].get(b, 0)
            if da > db:
                counts[a] += 1
            elif db > da:
                counts[b] += 1
            else:
                counts["Tie"] += 1
        outcomes = [a, b, "Tie"]
        key = f"Round 1 Longest HR - {a} v {b}"
        market[key] = _probs_to_market(outcomes, counts, n)
    return market


def _market_longest_hr_player(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        if s["derby_longest_player"]:
            counts[s["derby_longest_player"]] += 1
    return {"Player to Hit Longest HR": _probs_to_market(names, counts, n)}


def _market_longest_hr_distance(sims, n):
    vals = [s["derby_longest_dist"] for s in sims]
    return {"Length of Longest HR": _over_under_lines(vals)}


def _market_player_longest_distance(names, sims, n):
    market = {}
    for nm in names:
        vals = [s["r1_longest"].get(nm, 0) for s in sims if s["r1_longest"].get(nm, 0) > 0]
        if vals:
            market[f"{nm} Distance of Longest HR"] = _over_under_lines(vals)
    return market


def _market_record_broken(sims, n):
    count = sum(1 for s in sims if s["record_broken"])
    prob  = count / n
    return {"Will the Distance Record be Broken?": {
        "Yes": _fmt(prob),
        "No":  _fmt(1 - prob),
    }}


# ── exit velocity markets ──────────────────────────────────────

def _market_highest_ev_player(names, sims, n):
    counts = defaultdict(int)
    for s in sims:
        if s["derby_ev_player"]:
            counts[s["derby_ev_player"]] += 1
    return {"Player with Highest Exit Velocity HR": _probs_to_market(names, counts, n)}


def _market_highest_ev_value(sims, n):
    vals = [s["derby_max_ev"] for s in sims]
    return {"Highest Exit Velocity HR": _over_under_lines(vals)}


def _market_player_highest_ev(names, sims, n):
    market = {}
    for nm in names:
        vals = [s["player_max_ev"].get(nm, 0) for s in sims if s["player_max_ev"].get(nm, 0) > 0]
        if vals:
            market[f"{nm} Highest Exit Velocity HR"] = _over_under_lines(vals)
    return market


# ── category markets ───────────────────────────────────────────

def _market_winner_league(sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["winner_league"]] += 1
    return {"League of Winner": _probs_to_market(["AL", "NL"], counts, n)}


def _market_league_most_hrs(sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["league_most_hrs"]] += 1
    return {"League with Most Derby HRs": _probs_to_market(["AL", "NL"], counts, n)}


def _market_league_total_hrs(sims, n):
    al_vals = [s["league_hrs"]["AL"] for s in sims]
    nl_vals = [s["league_hrs"]["NL"] for s in sims]
    return {
        "AL Total HRs": _over_under_lines(al_vals),
        "NL Total HRs": _over_under_lines(nl_vals),
    }


def _market_winner_nationality(sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["winner_nationality"]] += 1
    return {"Winner - USA v Rest of the World": _probs_to_market(["USA", "International"], counts, n)}


def _market_winner_handedness(sims, n):
    counts = defaultdict(int)
    for s in sims:
        counts[s["winner_bats"]] += 1
    return {"Winner - Batter Handedness": _probs_to_market(["R", "L"], counts, n)}


# ── formatting helpers ─────────────────────────────────────────

def _probs_to_market(outcomes, counts, n):
    result = {}
    for o in outcomes:
        prob = counts.get(o, 0) / n
        result[str(o)] = _fmt(prob)
    return result


def _fmt(prob: float) -> dict:
    prob = max(1e-6, min(1 - 1e-6, prob))
    return {
        "prob":     round(prob, 6),
        "american": _to_american(prob),
    }


def _to_american(prob: float) -> str:
    if prob >= 0.5:
        odds = round(-(prob / (1 - prob)) * 100)
        return str(int(odds))
    else:
        odds = round(((1 - prob) / prob) * 100)
        return f"+{int(odds)}"


def _over_under_lines(vals: list) -> dict:
    """
    Generate Over/Under lines at the median and at ±5/±10 increments.
    Returns dict of { 'Over X.5': {prob, american}, 'Under X.5': {...} }
    """
    arr    = np.array(vals)
    median = float(np.median(arr))
    lines  = _generate_lines(median)
    result = {}
    for line in lines:
        over_prob  = float((arr > line).mean())
        under_prob = 1 - over_prob
        label_over  = f"Over {line}"
        label_under = f"Under {line}"
        result[label_over]  = _fmt(over_prob)
        result[label_under] = _fmt(under_prob)
    return result


def _generate_lines(median: float) -> list:
    """Return a spread of .5 lines around the median."""
    base = round(median)
    candidates = [base - 10, base - 5, base - 2, base - 1,
                  base, base + 1, base + 2, base + 5, base + 10]
    return [c + 0.5 for c in candidates if c >= 0]
