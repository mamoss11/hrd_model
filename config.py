# ─────────────────────────────────────────────────────────────
#  HRD Model — Configuration
# ─────────────────────────────────────────────────────────────

# ── Simulation ──────────────────────────────────────────────
N_SIMULATIONS = 100_000

# ── Derby Format (2026 rules) ────────────────────────────────
# Swing-based format: fixed swings per round, no timer, no bonus time.
# Hot hand rule: if a player homers on their last swing, they keep
# swinging until they do not hit a homer.
# Tiebreakers: R1 = longest HR distance; R2/Finals = 3-swing swing-offs.
DERBY_FORMAT = {
    "round_1": {
        "swings": 20,
        "players_advancing": 4,
    },
    "semifinals": {
        "swings": 15,
        "players_advancing": 2,
    },
    "finals": {
        "swings": 15,
        "players_advancing": 1,
    },
}

# Seeding for semis: 1v4, 2v3 by Round 1 HR rank
SEMIFINAL_SEEDING = [(0, 3), (1, 2)]   # index into sorted round-1 results

# ── Attribute Weights ────────────────────────────────────────
# Must sum to 1.0
ATTRIBUTE_WEIGHTS = {
    "bat_speed":          0.25,   # +0.05 from hr_per_pa reduction
    "max_exit_velo":      0.20,   # +0.05 from hr_per_pa reduction
    "pulled_barrel_pct":  0.20,
    "hr_per_pa":          0.15,   # lowered from 0.30 — reduces penalty on limited-PA guys
    "pct90_exit_velo":    0.20,   # +0.05 from hr_per_pa reduction
}

# ── HR Probability Calibration ───────────────────────────────
# Power score 0→1 maps linearly to HR probability per swing
# Bounds derived at runtime from hrd_outs_format.csv (see player_model.py).
# These fallback values are used only if the CSV is missing.
HR_PROB_MIN = 0.36   # tuned floor — calibrated against 2025 market shape; floors P20 result
HR_PROB_MAX = 0.50   # ceiling fallback; not currently binding (P85 = 0.4737)

# Percentiles of Round 1 hr_per_swing used to set the bounds.
# P15/P85 discards extreme outliers while capturing the realistic
# skill range of a competitive HRD field.
OUTS_LO_PCT = 20.0
OUTS_HI_PCT = 85.0

# ── Performance Variance ─────────────────────────────────────
# Each simulation draws a random day-factor per player:
#   hr_prob_sim = hr_prob * Normal(1.0, PERFORMANCE_STD)
# Models pitcher quality, feel on the day, crowd pressure etc.
# Higher value = more upsets, flatter outright prices.
# Typical useful range: 0.10 (low variance) to 0.25 (high variance)
PERFORMANCE_STD = 0.15
PERFORMANCE_CLIP = (0.50, 1.60)   # bounds on the multiplier

# ── Park Factors ─────────────────────────────────────────────
# Set HRD_VENUE to the host stadium name below.
# Use "Neutral" if the venue is unknown or not listed.
# hr_factor_R:   multiplier on per-swing HR probability for right-handed batters
# hr_factor_L:   multiplier on per-swing HR probability for left-handed batters
# dist_bonus_ft: feet added to every simulated HR distance (altitude / air density)
# Switch hitters use the average of R and L factors.
HRD_VENUE = "Citizens Bank Park"

PARK_FACTORS = {
    # ── Hitter-friendly ──────────────────────────────────────
    "Coors Field":              {"hr_factor_R": 1.20, "hr_factor_L": 1.20, "dist_bonus_ft":  8.0},  # ~5280ft elevation
    "Great American Ball Park": {"hr_factor_R": 1.13, "hr_factor_L": 1.16, "dist_bonus_ft":  2.0},  # ~482ft elevation; short RF/LF
    "American Family Field":    {"hr_factor_R": 1.09, "hr_factor_L": 1.07, "dist_bonus_ft":  0.0},  # MIL; retractable roof; cozy dimensions
    "Citizens Bank Park":       {"hr_factor_R": 1.09, "hr_factor_L": 1.11, "dist_bonus_ft":  0.0},  # PHI; 2026 HRD venue; short RF power alley
    "Guaranteed Rate Field":    {"hr_factor_R": 1.08, "hr_factor_L": 1.05, "dist_bonus_ft":  0.0},  # CWS; small dimensions
    "Globe Life Field":         {"hr_factor_R": 1.07, "hr_factor_L": 1.06, "dist_bonus_ft":  0.0},  # TEX; retractable roof; modern HR-friendly
    "Minute Maid Park":         {"hr_factor_R": 1.06, "hr_factor_L": 0.97, "dist_bonus_ft":  0.0},  # HOU; Crawford Boxes (315ft LF) favor RHB; deep RF suppresses LHB
    "Wrigley Field":            {"hr_factor_R": 1.06, "hr_factor_L": 1.04, "dist_bonus_ft":  0.0},  # CHI; wind-dependent; favors both hands on average
    "Yankee Stadium":           {"hr_factor_R": 1.05, "hr_factor_L": 1.18, "dist_bonus_ft":  1.0},  # NYY; short RF porch (314ft) strongly favors LHB
    "Chase Field":              {"hr_factor_R": 1.05, "hr_factor_L": 1.07, "dist_bonus_ft":  1.0},  # ARI; ~1100ft elevation; retractable roof
    # ── Moderate / near-neutral ──────────────────────────────
    "Truist Park":              {"hr_factor_R": 1.04, "hr_factor_L": 1.06, "dist_bonus_ft":  0.0},  # ATL; modern; slight LHB edge
    "Camden Yards":             {"hr_factor_R": 1.04, "hr_factor_L": 1.07, "dist_bonus_ft":  0.0},  # BAL; classic; short RF (318ft)
    "Rogers Centre":            {"hr_factor_R": 1.03, "hr_factor_L": 1.03, "dist_bonus_ft":  0.0},  # TOR; indoor dome; symmetric
    "Fenway Park":              {"hr_factor_R": 1.03, "hr_factor_L": 0.93, "dist_bonus_ft":  0.0},  # BOS; Pesky's Pole (302ft RF) helps RHB; Green Monster kills LHB HRs
    "Target Field":             {"hr_factor_R": 1.01, "hr_factor_L": 1.03, "dist_bonus_ft":  0.0},  # MIN; cold early season; modest LHB edge (short LF)
    "Progressive Field":        {"hr_factor_R": 1.00, "hr_factor_L": 0.98, "dist_bonus_ft":  0.0},  # CLE; fences moved in 2015; now near-neutral
    "Nationals Park":           {"hr_factor_R": 1.02, "hr_factor_L": 1.00, "dist_bonus_ft":  0.0},  # WAS; symmetric; mild RHB edge
    "Kauffman Stadium":         {"hr_factor_R": 0.99, "hr_factor_L": 0.95, "dist_bonus_ft":  0.0},  # KC; large dimensions; suppresses LHB more
    "Sutter Health Park":       {"hr_factor_R": 0.99, "hr_factor_L": 0.99, "dist_bonus_ft":  0.0},  # OAK (2025–); converted MiLB park; treat near-neutral
    "Tropicana Field":          {"hr_factor_R": 0.94, "hr_factor_L": 0.93, "dist_bonus_ft":  0.0},  # TB; indoor dome; cavernous outfield
    # ── Pitcher-friendly ─────────────────────────────────────
    "Busch Stadium":            {"hr_factor_R": 0.97, "hr_factor_L": 0.97, "dist_bonus_ft":  0.0},  # STL; large symmetric; suppresses both hands equally
    "Dodger Stadium":           {"hr_factor_R": 0.94, "hr_factor_L": 0.97, "dist_bonus_ft":  0.0},  # LAD; marine layer at night; suppresses fly balls
    "PNC Park":                 {"hr_factor_R": 0.93, "hr_factor_L": 0.91, "dist_bonus_ft":  0.0},  # PIT; deep CF/LCF; strong pitcher's park
    "Angel Stadium":            {"hr_factor_R": 0.93, "hr_factor_L": 0.95, "dist_bonus_ft":  0.0},  # LAA; large dimensions; deep CF (396ft)
    "Citi Field":               {"hr_factor_R": 0.93, "hr_factor_L": 0.96, "dist_bonus_ft":  0.0},  # NYM; fences moved in; still suppresses RHB
    "T-Mobile Park":            {"hr_factor_R": 0.96, "hr_factor_L": 0.90, "dist_bonus_ft":  0.0},  # SEA; marine air; strongly suppresses LHB
    "Comerica Park":            {"hr_factor_R": 0.90, "hr_factor_L": 0.88, "dist_bonus_ft":  0.0},  # DET; deep CF (420ft) and LCF; one of the toughest
    "loanDepot park":           {"hr_factor_R": 0.89, "hr_factor_L": 0.91, "dist_bonus_ft":  0.0},  # MIA; large retractable-roof park; deep CF (407ft)
    "Petco Park":               {"hr_factor_R": 0.85, "hr_factor_L": 0.92, "dist_bonus_ft":  0.0},  # SD; ocean air; suppresses RHB more due to deep RF
    "Oracle Park":              {"hr_factor_R": 0.78, "hr_factor_L": 0.87, "dist_bonus_ft": -3.0},  # SF; McCovey Cove RF (399ft+); cold ocean air; toughest park
    # ── Fallback ──────────────────────────────────────────────
    "Neutral":                  {"hr_factor_R": 1.00, "hr_factor_L": 1.00, "dist_bonus_ft":  0.0},
}

# ── Distance Model (feet) ────────────────────────────────────
# mean_dist = DIST_BASE + (max_exit_velo - DIST_EV_BASE) * DIST_EV_SCALE
#             + (bat_speed - DIST_BS_BASE) * DIST_BS_SCALE
DIST_BASE     = 375.0
DIST_EV_BASE  = 95.0
DIST_EV_SCALE = 3.2
DIST_BS_BASE  = 70.0
DIST_BS_SCALE = 0.5
DIST_STD      = 12.0          # std dev of individual HR distances
DIST_RECORD   = 504.0         # existing HRD distance record (feet)

# ── Exit Velocity Model ──────────────────────────────────────
# Derby EV per HR ~ Normal(max_ev * EV_MEAN_FACTOR, EV_STD)
EV_MEAN_FACTOR = 0.92
EV_STD         = 3.5

# ── Historical HRD Performance Weights ──────────────────────
# Pre-2024 data discounted due to format change
HISTORICAL_YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 1.0,
    2023: 0.40,
    2022: 0.30,
    2021: 0.25,
    2019: 0.20,
}

# Historical adjustment: past derby HR rate vs model prediction
# Keyed by mlb_id, values are round-1 HR totals and years under the current format.
# Add entries as data becomes available.
HISTORICAL_DERBY = {
    # Matt Olson: 2019 HRD (old format — low weight)
    621566: {"r1_hrs": [9], "years": [2019]},
}

# ── Players — 2026 HRD Field ─────────────────────────────────
PLAYERS = [
    {"name": "Kyle Schwarber",    "mlb_id": 656941, "team": "PHI",
     "league": "NL", "nationality": "USA",           "bats": "L"},
    {"name": "Junior Caminero",   "mlb_id": 691406, "team": "TB",
     "league": "AL", "nationality": "International", "bats": "R"},
    {"name": "Bryce Harper",      "mlb_id": 547180, "team": "PHI",
     "league": "NL", "nationality": "USA",           "bats": "L"},
    {"name": "Ben Rice",          "mlb_id": 700250, "team": "NYY",
     "league": "AL", "nationality": "USA",           "bats": "L"},
    {"name": "Willson Contreras", "mlb_id": 575929, "team": "BOS",
     "league": "AL", "nationality": "International", "bats": "R"},
    {"name": "Jordan Walker",     "mlb_id": 691023, "team": "STL",
     "league": "NL", "nationality": "USA",           "bats": "R"},
    {"name": "Jac Caglianone",    "mlb_id": 695506, "team": "KC",
     "league": "AL", "nationality": "USA",           "bats": "L"},
    {"name": "Munetaka Murakami", "mlb_id": 808959, "team": "CWS",
     "league": "AL", "nationality": "International", "bats": "L"},
]

# ── Statcast Season Window ───────────────────────────────────
SEASON_START = "2025-03-27"
SEASON_END   = "2026-07-11"
