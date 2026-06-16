# ─────────────────────────────────────────────────────────────
#  MLB Stats API — player name search and detail lookup
# ─────────────────────────────────────────────────────────────
import requests

_BASE    = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10

# AL = league id 103, NL = 104
_LEAGUE_MAP = {103: "AL", 104: "NL"}


def load_roster(season: int = 2026) -> list:
    """
    Fetch all active MLB players for the season in one call.
    Returns a list of dicts sorted by fullName: [{id, fullName}, ...]
    """
    resp = requests.get(
        f"{_BASE}/sports/1/players",
        params={"season": season, "gameType": "R"},
        timeout=20,
    )
    resp.raise_for_status()
    people = resp.json().get("people", [])
    return sorted(
        [{"id": p["id"], "fullName": p["fullName"]} for p in people],
        key=lambda x: x["fullName"],
    )


def search_players(name: str) -> list:
    """
    Search active MLB players by name.
    Returns a list of dicts with 'id' and 'fullName'.
    """
    resp = requests.get(
        f"{_BASE}/people/search",
        params={"names": name, "sportId": 1, "active": True},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("people", [])


def get_player_info(mlb_id: int) -> dict | None:
    """
    Fetch full details for one player by MLB ID.
    Makes two API calls: people (for bats/nationality) + teams (for abbr/league).
    Returns dict with: name, mlb_id, team, league, bats, nationality.
    Returns None if the player can't be found.
    """
    # ── 1. Player details ──────────────────────────────────────
    resp = requests.get(
        f"{_BASE}/people/{mlb_id}",
        params={"hydrate": "currentTeam"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    people = resp.json().get("people", [])
    if not people:
        return None

    p         = people[0]
    team_stub = p.get("currentTeam", {})
    team_id   = team_stub.get("id")

    # Bats: API returns 'R', 'L', or 'S' (switch). Model only uses R/L.
    bats_code = p.get("batSide", {}).get("code", "R")
    bats      = "L" if bats_code in ("L", "S") else "R"   # switch → L

    birth_country = p.get("birthCountry", "")
    nationality   = "USA" if birth_country == "USA" else "International"

    # ── 2. Team details (abbreviation + league) ────────────────
    team_abbr  = ""
    league     = ""

    if team_id:
        team_resp = requests.get(f"{_BASE}/teams/{team_id}", timeout=_TIMEOUT)
        if team_resp.ok:
            teams = team_resp.json().get("teams", [])
            if teams:
                t         = teams[0]
                team_abbr = t.get("abbreviation", "")
                league_id = t.get("league", {}).get("id")
                league    = _LEAGUE_MAP.get(league_id, "")
                # Fallback: parse league name string
                if not league:
                    league_name = t.get("league", {}).get("name", "")
                    if "American" in league_name:
                        league = "AL"
                    elif "National" in league_name:
                        league = "NL"

    return {
        "name":        p.get("fullName", ""),
        "mlb_id":      p.get("id"),
        "team":        team_abbr,
        "league":      league,
        "bats":        bats,
        "nationality": nationality,
        "is_switch":   bats_code == "S",   # flag so app can show a note
    }
