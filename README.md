# HRD Betting Model

Monte Carlo simulation model that prices betting markets for the MLB Home Run Derby.
Covers 30+ markets: outright winner, make the final, player props, H2H, totals, distance, exit velocity, and category bets.

---

## Setup

**Python 3.10+ required.**

```bash
git clone <repo-url>
cd hrd_model
pip install -r requirements.txt
```

---

## Running the app

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

The app defaults to **Load from cache** (pre-built player data — instant load).
Switch to **Fetch Statcast** in the sidebar if you want live data from Baseball Savant.

---

## Refreshing player data

When the field changes or you want fresh Statcast numbers, rebuild the cache:

```bash
python build_cache.py
```

This fetches the 8 configured players from Baseball Savant (~2-4 minutes on first run,
faster on subsequent runs due to local disk caching) and saves `data/player_attrs.json`.
Commit that file so coworkers get the updated data on next pull.

---

## CLI usage

```bash
python main.py                          # price all markets, print to terminal
python main.py --output results.json    # save output to JSON
python main.py --demo                   # run with placeholder players
```

---

## Configuring the field

Edit `config.py`:

- `PLAYERS` — the 8 contestants (name, MLB ID, team, league, handedness, nationality)
- `HRD_VENUE` — host stadium (park factors applied automatically)
- `ATTRIBUTE_WEIGHTS` — how Statcast metrics are weighted in the power score
- `HR_PROB_MIN` / `HR_PROB_MAX` — per-swing HR probability range
- `PERFORMANCE_STD` — day-to-day variance (higher = more upsets, flatter prices)
- `SEASON_START` / `SEASON_END` — Statcast data window

After changing `PLAYERS` or season dates, re-run `build_cache.py`.

---

## Other scripts

| Script | Purpose |
|--------|---------|
| `backtest.py` | Run model on 2022-2025 historical fields |
| `validate_2025.py` | Compare 2025 model output vs closing market lines |
| `winner_analysis.py` | Historical winner profile analysis |
| `calibration.py` | Tune HR_PROB_MIN/MAX against actual R1 HR data |

---

## Notes

- Odds are **true prices with no vig** — apply your own margin before publishing.
- Park factors are pre-set for all 30 MLB stadiums in `config.py`.
- The 2026 HRD venue is Citizens Bank Park (Philadelphia).
