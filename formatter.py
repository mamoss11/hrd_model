# ─────────────────────────────────────────────────────────────
#  HRD Model — Output Formatter
# ─────────────────────────────────────────────────────────────
import json


def print_markets(markets: dict, min_prob: float = 0.0001) -> None:
    """Pretty-print all priced markets to stdout."""
    for market_name, outcomes in sorted(markets.items()):
        print(f"\n{'-' * 60}")
        print(f"  {market_name}")
        print(f"{'-' * 60}")
        if isinstance(outcomes, dict):
            _print_outcomes(outcomes, min_prob)


def _print_outcomes(outcomes: dict, min_prob: float) -> None:
    # Determine if this is a direct {label: {prob, american}} or nested
    sample = next(iter(outcomes.values()))
    if isinstance(sample, dict) and "prob" in sample:
        # Flat market
        sorted_items = sorted(outcomes.items(), key=lambda x: -x[1]["prob"])
        for label, data in sorted_items:
            if data["prob"] < min_prob:
                continue
            pct  = f"{data['prob'] * 100:6.2f}%"
            odds = data["american"]
            print(f"  {str(label):<45}  {pct}  {odds:>8}")
    else:
        # Nested (Over/Under dict or per-player dict of dicts)
        for sub_label, sub_outcomes in outcomes.items():
            print(f"\n  [{sub_label}]")
            _print_outcomes(sub_outcomes, min_prob)


def save_markets_json(markets: dict, filepath: str) -> None:
    """Serialise markets dict to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(markets, f, indent=2, default=str)
    print(f"\nMarkets saved to {filepath}")
