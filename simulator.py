# ─────────────────────────────────────────────────────────────
#  HRD Model — Monte Carlo Simulation Engine
# ─────────────────────────────────────────────────────────────
import numpy as np

from config import (DERBY_FORMAT, SEMIFINAL_SEEDING, N_SIMULATIONS,
                    DIST_RECORD, PERFORMANCE_STD, PERFORMANCE_CLIP,
                    HRD_VENUE, PARK_FACTORS)

# Resolve park factors once at import time
_PARK             = PARK_FACTORS.get(HRD_VENUE, PARK_FACTORS["Neutral"])
_PARK_HR_FACTOR_R = _PARK["hr_factor_R"]
_PARK_HR_FACTOR_L = _PARK["hr_factor_L"]
_PARK_DIST_BONUS  = _PARK["dist_bonus_ft"]


# ── public entry point ─────────────────────────────────────────

def run_simulations(players: list, n: int = N_SIMULATIONS) -> list:
    """
    Run n full derby simulations.
    Returns a list of result dicts, one per simulation.
    """
    rng     = np.random.default_rng(seed=42)
    results = []
    for _ in range(n):
        results.append(_simulate_derby(players, rng))
    return results


# ── single derby simulation ────────────────────────────────────

def _simulate_derby(players: list, rng: np.random.Generator) -> dict:
    names = [p["name"] for p in players]

    # ── Round 1 ───────────────────────────────────────────────
    r1 = {p["name"]: _simulate_round(p, "round_1", rng) for p in players}

    r1_hrs    = {n: r1[n]["hrs"]     for n in names}
    r1_ranked = sorted(names, key=lambda n: r1_hrs[n], reverse=True)

    # Resolve ties at the 4th/5th boundary — R1 uses longest HR distance
    swing_offs = 0
    r1_longest = {n: r1[n]["max_dist"] for n in names}
    r1_ranked, so = _resolve_cutline(r1_ranked, r1_hrs, cut=4, rng=rng, distances=r1_longest)
    swing_offs += so

    semis_names   = r1_ranked[:4]
    semis_players = {p["name"]: p for p in players if p["name"] in semis_names}

    # ── Semifinals ────────────────────────────────────────────
    # Seeding: 1v4, 2v3 (by round-1 rank)
    semi_matchups = [
        (semis_names[s[0]], semis_names[s[1]])
        for s in SEMIFINAL_SEEDING
    ]
    semi_results  = {}
    finalists     = []

    for (a, b) in semi_matchups:
        ra = _simulate_round(semis_players[a], "semifinals", rng)
        rb = _simulate_round(semis_players[b], "semifinals", rng)
        semi_results[a] = ra
        semi_results[b] = rb

        if ra["hrs"] > rb["hrs"]:
            winner, loser = a, b
        elif rb["hrs"] > ra["hrs"]:
            winner, loser = b, a
        else:
            winner, loser, so = _swing_off(a, b, semis_players[a], semis_players[b], rng)
            swing_offs += so

        finalists.append(winner)

    # ── Final ─────────────────────────────────────────────────
    fa_p  = semis_players[finalists[0]]
    fb_p  = semis_players[finalists[1]]
    r_fa  = _simulate_round(fa_p, "finals", rng)
    r_fb  = _simulate_round(fb_p, "finals", rng)

    if r_fa["hrs"] > r_fb["hrs"]:
        champion = finalists[0]
    elif r_fb["hrs"] > r_fa["hrs"]:
        champion = finalists[1]
    else:
        champion, _, so = _swing_off(finalists[0], finalists[1], fa_p, fb_p, rng)
        swing_offs += so

    # ── Aggregate ─────────────────────────────────────────────
    champion_player = next(p for p in players if p["name"] == champion)

    # Total HRs for each player across all rounds they played
    all_round_hrs = {}
    for nm in names:
        hrs = r1[nm]["hrs"]
        if nm in semi_results:
            hrs += semi_results[nm]["hrs"]
        if nm == finalists[0]:
            hrs += r_fa["hrs"]
        elif nm == finalists[1]:
            hrs += r_fb["hrs"]
        all_round_hrs[nm] = hrs

    winner_total_hrs = all_round_hrs[champion]

    # Per-player max dist and EV — running max across all rounds played
    max_dist_by_player = {nm: r1[nm]["max_dist"] for nm in names}
    max_ev_by_player   = {nm: r1[nm]["max_ev"]   for nm in names}

    for nm, rd in semi_results.items():
        if rd["max_dist"] > max_dist_by_player[nm]:
            max_dist_by_player[nm] = rd["max_dist"]
        if rd["max_ev"] > max_ev_by_player[nm]:
            max_ev_by_player[nm] = rd["max_ev"]

    for nm, rd in [(finalists[0], r_fa), (finalists[1], r_fb)]:
        if rd["max_dist"] > max_dist_by_player[nm]:
            max_dist_by_player[nm] = rd["max_dist"]
        if rd["max_ev"] > max_ev_by_player[nm]:
            max_ev_by_player[nm] = rd["max_ev"]

    # r1_longest already computed above for distance tiebreaker

    # Global stats
    derby_longest_dist   = max(max_dist_by_player.values())
    derby_longest_player = max(max_dist_by_player, key=max_dist_by_player.get)
    derby_max_ev         = max(max_ev_by_player.values())
    derby_ev_player      = max(max_ev_by_player, key=max_ev_by_player.get)

    # Per-player max EV
    player_max_ev = max_ev_by_player

    # AL/NL league totals
    player_map = {p["name"]: p for p in players}
    league_hrs = {"AL": 0, "NL": 0}
    for nm, hrs in all_round_hrs.items():
        lg = player_map[nm]["league"]
        if lg in league_hrs:
            league_hrs[lg] += hrs

    return {
        "winner":               champion,
        "winner_league":        champion_player["league"],
        "winner_nationality":   champion_player["nationality"],
        "winner_bats":          champion_player["bats"],
        "finalists":            tuple(sorted(finalists)),   # sorted for set-based lookup
        "finalists_ordered":    tuple(finalists),           # [semi1_winner, semi2_winner]
        "champion_beats":       (champion, [f for f in finalists if f != champion][0]),
        "semifinalists":        tuple(sorted(semis_names)),
        "semi_matchups":        tuple(tuple(sorted(m)) for m in semi_matchups),
        "r1_hrs":               r1_hrs,
        "r1_ranked":            r1_ranked,
        "r1_longest":           r1_longest,
        "all_round_hrs":        all_round_hrs,
        "winner_total_hrs":     winner_total_hrs,
        "total_derby_hrs":      sum(all_round_hrs.values()),
        "total_r1_hrs":         sum(r1_hrs.values()),
        "most_r1_hrs_player":   max(r1_hrs, key=r1_hrs.get),
        "most_r1_hrs_count":    max(r1_hrs.values()),
        "derby_longest_dist":   derby_longest_dist,
        "derby_longest_player": derby_longest_player,
        "derby_max_ev":         derby_max_ev,
        "derby_ev_player":      derby_ev_player,
        "player_max_ev":        player_max_ev,
        "league_hrs":           league_hrs,
        "league_most_hrs":      max(league_hrs, key=league_hrs.get),
        "swing_offs":           swing_offs,
        "record_broken":        derby_longest_dist > DIST_RECORD,
    }


# ── round simulation ───────────────────────────────────────────

def _simulate_round(player: dict, round_key: str, rng: np.random.Generator) -> dict:
    fmt = DERBY_FORMAT[round_key]

    # Fresh pitcher factor every round — models how well the pitcher
    # locates pitches in that specific session (fatigue, consistency, etc.)
    pitcher_factor = float(np.clip(
        rng.normal(1.0, PERFORMANCE_STD), PERFORMANCE_CLIP[0], PERFORMANCE_CLIP[1]
    ))

    # Park factor — handedness-specific
    bats = player.get("bats", "R")
    if bats == "L":
        park_factor = _PARK_HR_FACTOR_L
    elif bats == "R":
        park_factor = _PARK_HR_FACTOR_R
    else:  # switch hitter
        park_factor = (_PARK_HR_FACTOR_L + _PARK_HR_FACTOR_R) / 2.0

    hr_prob = float(np.clip(player["hr_prob"] * pitcher_factor * park_factor, 0.05, 0.95))

    # Fixed swing allotment
    swings = fmt["swings"]
    hits   = rng.random(swings) < hr_prob
    hrs    = int(hits.sum())

    # Hot hand: if last swing was a HR, keep swinging until non-HR
    if hits[-1]:
        while True:
            if rng.random() < hr_prob:
                hrs += 1
            else:
                break

    # Max distance and EV for this round (only max is needed downstream)
    if hrs > 0:
        raw_dists = rng.normal(player["mean_hr_dist"] + _PARK_DIST_BONUS, player["dist_std"], hrs)
        max_dist  = float(np.clip(raw_dists, 300, 600).max())

        raw_evs = rng.normal(player["mean_hr_ev"], player["ev_std"], hrs)
        max_ev  = float(np.clip(raw_evs, 90, player["max_exit_velo"] * 1.03).max())
    else:
        max_dist = 0.0
        max_ev   = 0.0

    return {"hrs": hrs, "swings": swings, "max_dist": max_dist, "max_ev": max_ev}


# ── swing-off tiebreaker ───────────────────────────────────────

def _swing_off(name_a: str, name_b: str, p_a: dict, p_b: dict,
               rng: np.random.Generator) -> tuple:
    """
    Simulate a 3-swing swing-off. Repeat until broken.
    Returns (winner_name, loser_name, n_swingoffs_used).
    """
    n_so = 0
    while True:
        n_so  += 1
        hrs_a  = int((rng.random(3) < p_a["hr_prob"]).sum())
        hrs_b  = int((rng.random(3) < p_b["hr_prob"]).sum())
        if hrs_a != hrs_b:
            if hrs_a > hrs_b:
                return name_a, name_b, n_so
            else:
                return name_b, name_a, n_so


def _resolve_cutline(ranked: list, hrs: dict, cut: int,
                     rng: np.random.Generator,
                     distances: dict = None) -> tuple:
    """
    Ensure no tie straddles the cut position.
    If distances provided, uses longest HR distance as tiebreaker (2026 R1 rule).
    Otherwise falls back to 3-swing swing-off (semis/finals).
    Returns (updated_ranking, n_swingoffs).
    """
    if len(ranked) <= cut:
        return ranked, 0

    cutline_score = hrs[ranked[cut - 1]]    # score of last qualifier
    next_score    = hrs[ranked[cut]]        # score of first non-qualifier
    if cutline_score != next_score:
        return ranked, 0

    # Gather all tied players at the cutline
    tied   = [n for n in ranked if hrs[n] == cutline_score]
    others = [n for n in ranked if hrs[n] != cutline_score]

    if distances is not None:
        # 2026 R1 rule: longest HR distance breaks the tie
        tied = sorted(tied, key=lambda n: distances.get(n, 0.0), reverse=True)
        n_so = 0   # distance tiebreak is not a swing-off
    else:
        # Swing-off: random shuffle (each round handled separately)
        tied_arr = np.array(tied)
        rng.shuffle(tied_arr)
        tied = tied_arr.tolist()
        n_so = 1

    new_ranking = sorted(others, key=lambda n: hrs[n], reverse=True) + tied
    return new_ranking, n_so
