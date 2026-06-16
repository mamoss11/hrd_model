"""
Build a local Statcast attribute cache for the configured player field.

Run this once (and again whenever the field changes or you want fresh data):

    python build_cache.py

Saves data/player_attrs.json — the Streamlit app loads this by default
so coworkers can run the model without waiting for Baseball Savant fetches.
"""

import json
import os
import numpy as np
import config as cfg
from data_fetcher import load_player_attributes


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print(
        f"Fetching Statcast for {len(cfg.PLAYERS)} players "
        f"({cfg.SEASON_START} -> {cfg.SEASON_END})...\n"
    )
    players = load_player_attributes(cfg.PLAYERS, cfg.SEASON_START, cfg.SEASON_END)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "player_attrs.json")
    with open(out_path, "w") as f:
        json.dump(players, f, indent=2, cls=_NpEncoder)

    print(f"\nSaved {len(players)} players to {out_path}\n")
    print(f"{'Player':<22} {'BatSpd':>7} {'MaxEV':>7} {'HR/PA':>7}")
    print("-" * 48)
    for p in players:
        print(
            f"{p['name']:<22} "
            f"{p['bat_speed']:>7.1f} "
            f"{p['max_exit_velo']:>7.1f} "
            f"{p['hr_per_pa']:>7.3f}"
        )


if __name__ == "__main__":
    main()
