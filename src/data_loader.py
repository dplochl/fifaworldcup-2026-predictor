"""
data_loader.py
--------------
Data pipeline for the WC 2026 predictor.

Loads and preprocesses historical match data, computes Glicko-1 ratings,
and saves all outputs to data/processed/.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# WC 2026 Groups & Fixtures (hardcoded from official draw)
# ---------------------------------------------------------------------------

WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ── Official WC 2026 knockout bracket ─────────────────────────────────────────
# R32: 16 matches. Spec: ("W"|"R", group) = winner/runner-up of that group;
#                        ("T", slot_id)   = 3rd-place slot (id = match number).
WC2026_R32_BRACKET = [
    (73, ("R","A"), ("R","B")),  (74, ("W","E"), ("T",74)),
    (75, ("W","F"), ("R","C")),  (76, ("W","C"), ("R","F")),
    (77, ("W","I"), ("T",77)),   (78, ("R","E"), ("R","I")),
    (79, ("W","A"), ("T",79)),   (80, ("W","L"), ("T",80)),
    (81, ("W","D"), ("T",81)),   (82, ("W","G"), ("T",82)),
    (83, ("R","K"), ("R","L")),  (84, ("W","H"), ("R","J")),
    (85, ("W","B"), ("T",85)),   (86, ("W","J"), ("R","H")),
    (87, ("W","K"), ("T",87)),   (88, ("R","D"), ("R","G")),
]

# For each 3rd-place slot, which groups are eligible to fill it
# (guarantees no same-group rematch in R32).
WC2026_THIRD_PLACE_ELIGIBLE = {
    74: {"A","B","C","D","F"},  77: {"C","D","F","G","H"},
    79: {"C","E","F","H","I"},  80: {"E","H","I","J","K"},
    81: {"B","E","F","I","J"},  82: {"A","E","H","I","J"},
    85: {"E","F","G","I","J"},  87: {"D","E","I","J","L"},
}

# Bracket progression — indices are 0-based within WC2026_R32_BRACKET order.
# R16: which two R32 match-indices pair together
WC2026_R16_PAIRS = [(1,4),(0,2),(3,5),(6,7),(10,11),(8,9),(13,15),(12,14)]
# QF:  which two R16 match-indices pair together
WC2026_QF_PAIRS  = [(0,1),(4,5),(2,3),(6,7)]
# SF:  which two QF match-indices pair together
WC2026_SF_PAIRS  = [(0,1),(2,3)]

# Confederation offset applied at prediction time to correct for inflated Elo
# from teams that dominate weaker confederations.
# UEFA and CONMEBOL are the baseline (0). Others are adjusted down.
CONFEDERATION_OFFSETS = {
    "UEFA":     0,
    "CONMEBOL": 0,
    "CONCACAF": -100,
    "CAF":      -50,
    "AFC":      -80,
    "OFC":      -150,
}

TEAM_CONFEDERATION = {
    # UEFA
    "Spain": "UEFA", "France": "UEFA", "England": "UEFA", "Portugal": "UEFA",
    "Germany": "UEFA", "Belgium": "UEFA", "Netherlands": "UEFA", "Croatia": "UEFA",
    "Switzerland": "UEFA", "Austria": "UEFA", "Turkey": "UEFA", "Norway": "UEFA",
    "Sweden": "UEFA", "Czech Republic": "UEFA", "Scotland": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Italy": "UEFA", "Denmark": "UEFA",
    "Ukraine": "UEFA", "Poland": "UEFA", "Wales": "UEFA", "Serbia": "UEFA",
    "Romania": "UEFA", "Slovenia": "UEFA", "Hungary": "UEFA", "Greece": "UEFA",
    "Slovakia": "UEFA", "Russia": "UEFA", "Iceland": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Costa Rica": "CONCACAF", "Haiti": "CONCACAF",
    "Curacao": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "South Africa": "CAF", "Ivory Coast": "CAF",
    "DR Congo": "CAF", "Cameroon": "CAF", "Nigeria": "CAF", "Cape Verde": "CAF",
    "Mali": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC",
    "Jordan": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# Host nations — play group stage games at home, giving them genuine home advantage
HOST_NATIONS = {"United States", "Canada", "Mexico"}

TEAM_FLAGS = {
    "Argentina": "🇦🇷", "Spain": "🇪🇸", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Portugal": "🇵🇹", "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Germany": "🇩🇪", "Croatia": "🇭🇷", "Colombia": "🇨🇴",
    "Mexico": "🇲🇽", "Senegal": "🇸🇳", "United States": "🇺🇸", "Uruguay": "🇺🇾",
    "Japan": "🇯🇵", "Switzerland": "🇨🇭", "Iran": "🇮🇷", "Turkey": "🇹🇷",
    "Austria": "🇦🇹", "Ecuador": "🇪🇨", "South Korea": "🇰🇷", "Australia": "🇦🇺",
    "Algeria": "🇩🇿", "Egypt": "🇪🇬", "Canada": "🇨🇦", "Norway": "🇳🇴",
    "Ivory Coast": "🇨🇮", "Panama": "🇵🇦", "Sweden": "🇸🇪",
    "Czech Republic": "🇨🇿", "Paraguay": "🇵🇾", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "DR Congo": "🇨🇩", "Tunisia": "🇹🇳", "Uzbekistan": "🇺🇿",
    "Qatar": "🇶🇦", "Iraq": "🇮🇶", "South Africa": "🇿🇦",
    "Saudi Arabia": "🇸🇦", "Jordan": "🇯🇴", "Bosnia and Herzegovina": "🇧🇦",
    "Cape Verde": "🇨🇻", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Curacao": "🇨🇼", "New Zealand": "🇳🇿", "Italy": "🇮🇹",
    "Denmark": "🇩🇰", "Nigeria": "🇳🇬", "Ukraine": "🇺🇦",
    "Poland": "🇵🇱", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Serbia": "🇷🇸", "Cameroon": "🇨🇲",
    "Venezuela": "🇻🇪", "Chile": "🇨🇱", "Romania": "🇷🇴", "Peru": "🇵🇪",
    "Slovenia": "🇸🇮", "Iceland": "🇮🇸", "Hungary": "🇭🇺", "Greece": "🇬🇷",
    "Costa Rica": "🇨🇷", "Bolivia": "🇧🇴", "Russia": "🇷🇺", "Slovakia": "🇸🇰",
}

# FIFA rankings — last updated: 2026-06-06
# Includes non-WC teams so the match predictor works for any matchup
FIFA_RANKINGS = {
    # WC 2026 participants
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4,
    "Portugal": 5, "Brazil": 6, "Morocco": 7, "Netherlands": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11, "Colombia": 13,
    "Mexico": 14, "Senegal": 15, "United States": 16, "Uruguay": 17,
    "Japan": 18, "Switzerland": 19, "Iran": 20, "Turkey": 22,
    "Austria": 23, "Ecuador": 24, "South Korea": 25, "Australia": 27,
    "Algeria": 28, "Egypt": 29, "Canada": 30, "Norway": 31,
    "Ivory Coast": 33, "Panama": 34, "Sweden": 38,
    "Czech Republic": 39, "Paraguay": 40, "Scotland": 43,
    "DR Congo": 45, "Tunisia": 46, "Uzbekistan": 50,
    "Qatar": 54, "Iraq": 55, "South Africa": 60,
    "Saudi Arabia": 61, "Jordan": 63, "Bosnia and Herzegovina": 64,
    "Cape Verde": 68, "Ghana": 73, "Haiti": 80,
    "Curacao": 83, "New Zealand": 85,
    # Notable non-qualifiers (needed for match predictor & training data)
    "Italy": 12, "Denmark": 21, "Nigeria": 26, "Ukraine": 32,
    "Poland": 36, "Wales": 37, "Serbia": 42, "Cameroon": 44,
    "Venezuela": 49, "Chile": 53, "Romania": 56, "Peru": 57,
    "Slovenia": 58, "Iceland": 74, "Hungary": 41, "Greece": 47,
    "Costa Rica": 51, "Bolivia": 77, "Russia": 35, "Slovakia": 48,
}


def compute_recency_weight(date, reference_year: int = 2026, decay: float = 0.05) -> float:
    """
    Exponential decay so recent matches matter more than old ones.
    decay=0.05 → a match 10 years ago has weight exp(-0.5) ≈ 0.61 relative to today.
    """
    try:
        years_ago = max(reference_year - pd.Timestamp(date).year, 0)
        return float(np.exp(-decay * years_ago))
    except Exception:
        return 1.0


def assign_match_weight(tournament: str) -> float:
    """
    Assign a training weight based on how competitively meaningful a match is.
    Higher weight = optimizer learns more from this match.
    """
    t = str(tournament).lower()

    # Tier 1 — flagship tournaments
    if any(x in t for x in ["fifa world cup", "uefa euro", "copa américa", "copa america",
                              "african cup of nations", "afc asian cup", "gold cup",
                              "oceania nations cup"]):
        if "qualif" not in t:
            return 1.0

    # Tier 2 — secondary continental finals
    if any(x in t for x in ["confederations cup", "aff championship",
                              "eaff championship", "waff championship"]):
        return 0.75

    # Tier 3 — nations leagues
    if "nations league" in t:
        return 0.7

    # Tier 4 — world cup & major continental qualifiers
    if any(x in t for x in ["world cup qualification", "euro qualification",
                              "african cup of nations qualification",
                              "afc asian cup qualification", "copa america qualification",
                              "gold cup qualification"]):
        return 0.5

    # Tier 5 — minor regional tournaments & qualifiers
    if any(x in t for x in ["gulf cup", "saff", "cecafa", "cosafa", "cfu",
                              "uncaf", "arab cup", "afc challenge"]):
        return 0.3

    # Tier 6 — friendlies
    if "friendly" in t:
        return 0.2

    # Tier 7 — non-FIFA / novelty events
    if any(x in t for x in ["island games", "conifa"]):
        return 0.1

    return 0.4  # unknown: treat like a minor qualifier


def load_historical_results(filepath: str = None) -> pd.DataFrame:
    """
    Load international football results.
    If filepath is None, looks for data/raw/results.csv
    Download from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    """
    if filepath is None:
        filepath = RAW / "results.csv"

    if not Path(filepath).exists():
        print(f"[!] results.csv not found at {filepath}")
        print("    Download from Kaggle and place at data/raw/results.csv")
        print("    kaggle datasets download martj42/international-football-results-from-1872-to-2017")
        return None

    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"[+] Loaded {len(df):,} historical matches ({df['date'].min().year}–{df['date'].max().year})")
    return df


def filter_recent(df: pd.DataFrame, since_year: int = 2000) -> pd.DataFrame:
    """Keep only matches from a given year onwards (football has changed a lot)."""
    df = df[df["date"].dt.year >= since_year].copy()
    print(f"[+] Filtered to {len(df):,} matches since {since_year}")
    return df


def add_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add FIFA ranking columns for home and away teams."""
    df = df.copy()
    df["home_ranking"] = df["home_team"].map(FIFA_RANKINGS)
    df["away_ranking"] = df["away_team"].map(FIFA_RANKINGS)
    df["ranking_diff"] = df["home_ranking"] - df["away_ranking"]  # negative = home is better ranked
    return df


def compute_glicko_ratings(
    df: pd.DataFrame,
    initial_rating: float = 1500.0,
    initial_rd: float = 350.0,
    min_rd: float = 20.0,
    c_sq_per_day: float = 0.3,
) -> tuple:
    """
    Compute Glicko-1 ratings for all teams.

    Each team has two values:
      - rating : same role as Elo, updated based on results
      - RD     : rating deviation — uncertainty about the true rating.
                 Decreases as a team plays more, increases during inactivity.

    High RD means we're less certain → match outcomes are harder to predict
    → draw probability should be higher.

    c_sq_per_day: how fast RD grows per inactive day (RD² += c_sq_per_day * days).
    """
    q = np.log(10) / 400.0

    ratings: dict[str, float] = {}
    rds: dict[str, float] = {}
    last_played: dict[str, pd.Timestamp] = {}

    def get_r(t):  return ratings.get(t, initial_rating)
    def get_rd(t): return rds.get(t, initial_rd)

    def g(rd):
        return 1.0 / np.sqrt(1 + 3 * q**2 * rd**2 / np.pi**2)

    def E(r, r_opp, rd_opp):
        return 1.0 / (1 + 10 ** (-g(rd_opp) * (r - r_opp) / 400.0))

    def inactivity_rd(rd, days):
        return min(np.sqrt(rd**2 + c_sq_per_day * days), initial_rd)

    def glicko_update(r, rd, r_opp, rd_opp, score):
        g_opp = g(rd_opp)
        e = E(r, r_opp, rd_opp)
        d_sq = 1.0 / (q**2 * g_opp**2 * e * (1 - e))
        new_rd = max(1.0 / np.sqrt(1.0 / rd**2 + 1.0 / d_sq), min_rd)
        new_r  = r + (q / (1.0 / rd**2 + 1.0 / d_sq)) * g_opp * (score - e)
        return new_r, new_rd

    elo_home_list, elo_away_list = [], []
    rd_home_list,  rd_away_list  = [], []

    for _, row in df.sort_values("date").iterrows():
        h, a   = row["home_team"], row["away_team"]
        date   = row["date"]

        # Age up RD for inactivity before this match
        for team in (h, a):
            if team in last_played:
                days = (date - last_played[team]).days
                if days > 0:
                    rds[team] = inactivity_rd(get_rd(team), days)

        rh, ra   = get_r(h),  get_r(a)
        rdh, rda = get_rd(h), get_rd(a)

        elo_home_list.append(rh)
        elo_away_list.append(ra)
        rd_home_list.append(rdh)
        rd_away_list.append(rda)

        last_played[h] = date
        last_played[a] = date

        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue

        if row["home_score"] > row["away_score"]:   sh, sa = 1.0, 0.0
        elif row["home_score"] < row["away_score"]: sh, sa = 0.0, 1.0
        else:                                        sh, sa = 0.5, 0.5

        ratings[h], rds[h] = glicko_update(rh, rdh, ra, rda, sh)
        ratings[a], rds[a] = glicko_update(ra, rda, rh, rdh, sa)

    df = df.copy()
    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["rd_home"]  = rd_home_list
    df["rd_away"]  = rd_away_list

    # Save current ratings — include RD so simulator and dashboard can use it
    current = [
        {"team": t, "elo": ratings.get(t, initial_rating), "rd": rds.get(t, initial_rd)}
        for t in set(ratings) | set(rds)
    ]
    elo_df = pd.DataFrame(current).sort_values("elo", ascending=False)
    elo_df.to_csv(PROCESSED / "current_elo.csv", index=False)
    print(f"[+] Computed Glicko ratings for {len(ratings)} teams")

    return df, ratings



def compute_team_form(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute each team's rolling average goals scored over their last N matches.
    Form is recorded BEFORE each match so there's no data leakage.
    Also saves current_form.csv with up-to-date values for predictions.
    """
    df = df.sort_values("date").reset_index(drop=True)
    DEFAULT = 1.3  # global average goals per team per match

    history: dict[str, list] = defaultdict(list)

    home_form, away_form = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]

        home_form.append(np.mean(history[h][-window:]) if history[h] else DEFAULT)
        away_form.append(np.mean(history[a][-window:]) if history[a] else DEFAULT)

        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            history[h].append(row["home_score"])
            history[a].append(row["away_score"])

    df["form_home"] = home_form
    df["form_away"] = away_form

    # Save current form for each team (used at prediction time)
    current = [
        {"team": t, "form_scored": np.mean(v[-window:])}
        for t, v in history.items()
    ]
    form_df = pd.DataFrame(current)
    form_df.to_csv(PROCESSED / "current_form.csv", index=False)
    print(f"[+] Computed form stats for {len(form_df)} teams (window={window})")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for modelling.
    Each match produces TWO rows: one per team (home and away perspective).
    Target: goals scored by that team in the match.
    """
    rows = []

    for _, r in df.iterrows():
        tournament = r.get("tournament", "")
        weight = assign_match_weight(tournament) * compute_recency_weight(r["date"])
        is_neutral = bool(r["neutral"]) if pd.notna(r.get("neutral")) else False

        squad_diff  = r.get("squad_rating_home", 74.0) - r.get("squad_rating_away", 74.0)
        rd_combined = (r.get("rd_home", 350.0) + r.get("rd_away", 350.0)) / 2.0

        # Home team row
        rows.append({
            "team": r["home_team"],
            "opponent": r["away_team"],
            "goals_scored": r["home_score"],
            "goals_conceded": r["away_score"],
            "is_home": 0 if is_neutral else 1,
            "elo_diff": r["elo_diff"],
            "ranking_diff": r.get("ranking_diff", 0),
            "form_scored": r.get("form_home", 1.3),
            "squad_rating_diff": squad_diff,
            "rd_combined": rd_combined,
            "date": r["date"],
            "tournament": tournament,
            "match_weight": weight,
            "is_wc": "FIFA World Cup" in str(tournament),
        })
        # Away team row — never has home advantage
        rows.append({
            "team": r["away_team"],
            "opponent": r["home_team"],
            "goals_scored": r["away_score"],
            "goals_conceded": r["home_score"],
            "is_home": 0,
            "elo_diff": -r["elo_diff"],
            "ranking_diff": -r.get("ranking_diff", 0),
            "form_scored": r.get("form_away", 1.3),
            "squad_rating_diff": -squad_diff,
            "rd_combined": rd_combined,
            "date": r["date"],
            "tournament": tournament,
            "match_weight": weight,
            "is_wc": "FIFA World Cup" in str(tournament),
        })

    features = pd.DataFrame(rows)
    features.to_csv(PROCESSED / "features.csv", index=False)
    print(f"[+] Built feature matrix: {len(features):,} rows")
    return features


# EA FC uses different names for some nations
EA_FC_NAME_MAP = {
    "Korea Republic":      "South Korea",
    "Côte d'Ivoire":       "Ivory Coast",
    "Congo DR":            "DR Congo",
    "Cape Verde Islands":  "Cape Verde",
    "Bosnia & Herzegovina":"Bosnia and Herzegovina",
    "Türkiye":             "Turkey",
    "IR Iran":             "Iran",
    "Chinese Taipei":      "Taiwan",
}


def load_squad_ratings(ratings_dir: str = "data/raw/fifa_ratings", top_n: int = 23) -> pd.DataFrame:
    """
    Load EA FC/FIFA player ratings for all available editions.
    Supports two formats:
      - male_players.csv (FC 24 dataset): single file with a 'fifa_version' column
      - players_XX.csv (FIFA 22 dataset): one file per edition
    Computes average overall of the top N players per nationality per year.
    Saves current_squad_ratings.csv (latest year) for prediction time.
    """
    ratings_path = Path(ratings_dir)
    all_ratings = []

    # Prefer the comprehensive male_players.csv if available
    comprehensive = ratings_path / "male_players.csv"
    if comprehensive.exists():
        df = pd.read_csv(comprehensive, usecols=["fifa_version", "nationality_name", "overall"], low_memory=False)
        df["year"] = (2000 + df["fifa_version"].astype(int))
        for year, group in df.groupby("year"):
            team_ratings = (
                group.groupby("nationality_name")["overall"]
                .apply(lambda x: x.nlargest(top_n).mean())
                .reset_index()
            )
            team_ratings.columns = ["nationality_name", "avg_overall"]
            team_ratings["year"] = year
            team_ratings["team"] = team_ratings["nationality_name"].map(lambda n: EA_FC_NAME_MAP.get(n, n))
            all_ratings.append(team_ratings[["year", "team", "avg_overall"]])
    else:
        # Fallback: individual players_XX.csv files
        for filepath in sorted(ratings_path.glob("players_[0-9]*.csv")):
            year = 2000 + int(filepath.stem.split("_")[1])
            try:
                df = pd.read_csv(filepath, usecols=["nationality_name", "overall"], low_memory=False)
            except Exception:
                continue
            team_ratings = (
                df.groupby("nationality_name")["overall"]
                .apply(lambda x: x.nlargest(top_n).mean())
                .reset_index()
            )
            team_ratings.columns = ["nationality_name", "avg_overall"]
            team_ratings["year"] = year
            team_ratings["team"] = team_ratings["nationality_name"].map(lambda n: EA_FC_NAME_MAP.get(n, n))
            all_ratings.append(team_ratings[["year", "team", "avg_overall"]])

    if not all_ratings:
        print("[!] No FIFA rating files found — skipping squad ratings")
        return pd.DataFrame(columns=["year", "team", "avg_overall"])

    ratings_df = pd.concat(all_ratings, ignore_index=True)

    latest_year = ratings_df["year"].max()
    current = ratings_df[ratings_df["year"] == latest_year][["team", "avg_overall"]].copy()
    current.to_csv(PROCESSED / "current_squad_ratings.csv", index=False)
    print(f"[+] Loaded EA FC ratings {ratings_df['year'].min()}–{latest_year} "
          f"({ratings_df['team'].nunique()} teams covered)")

    return ratings_df


def add_squad_ratings(df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add squad_rating_home and squad_rating_away columns to the match dataframe.
    Uses the closest available FIFA edition year for each match.
    Falls back to the global average for teams/years with no data.
    """
    if ratings_df.empty:
        df["squad_rating_home"] = 74.0
        df["squad_rating_away"] = 74.0
        return df

    global_avg = ratings_df["avg_overall"].mean()
    available_years = sorted(ratings_df["year"].unique())
    lookup = {(r["year"], r["team"]): r["avg_overall"] for _, r in ratings_df.iterrows()}

    def get_rating(team: str, match_year: int) -> float:
        if match_year < available_years[0]:
            return global_avg
        year = max(y for y in available_years if y <= match_year)
        return lookup.get((year, team), global_avg)

    match_years = pd.to_datetime(df["date"]).dt.year
    df = df.copy()
    df["squad_rating_home"] = [get_rating(t, y) for t, y in zip(df["home_team"], match_years)]
    df["squad_rating_away"] = [get_rating(t, y) for t, y in zip(df["away_team"], match_years)]

    coverage = (match_years >= available_years[0]).mean() * 100
    print(f"[+] Added squad ratings ({coverage:.0f}% of matches have year coverage)")
    return df


def save_wc2026_fixtures() -> pd.DataFrame:
    """Save the WC 2026 group stage fixtures to CSV."""
    fixtures = []
    for group, teams in WC2026_GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                fixtures.append({
                    "group": group,
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "neutral": True,
                })
    df = pd.DataFrame(fixtures)
    df.to_csv(PROCESSED / "wc2026_fixtures.csv", index=False)
    print(f"[+] Saved {len(df)} WC 2026 group stage fixtures")
    return df


if __name__ == "__main__":
    print("=== WC 2026 Predictor — Data Pipeline ===\n")

    df = load_historical_results()
    if df is None:
        print("\nPlace results.csv in data/raw/ and re-run.")
    else:
        df = filter_recent(df, since_year=2000)
        df = add_fifa_rankings(df)
        df, elo = compute_glicko_ratings(df)
        df = compute_team_form(df)
        squad_ratings = load_squad_ratings()
        df = add_squad_ratings(df, squad_ratings)
        features = build_features(df)
        fixtures = save_wc2026_fixtures()
        df.to_csv(PROCESSED / "matches_processed.csv", index=False)
        print("\n✓ Pipeline complete. Files saved to data/processed/")
