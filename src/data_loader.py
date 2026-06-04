"""
data_loader.py
--------------
Downloads and preprocesses all data needed for the WC 2026 predictor.

Data sources:
  - Kaggle: international football results 1872–2024
  - eloratings.net: historical Elo ratings
  - WC 2026 group fixtures (hardcoded from official draw)
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# WC 2026 Groups & Fixtures (hardcoded from official draw)
# ---------------------------------------------------------------------------

WC2026_GROUPS = {
    "A": ["USA", "Panama", "Algeria", "Morocco"],
    "B": ["Argentina", "Chile", "Peru", "Uzbekistan"],
    "C": ["Brazil", "Mexico", "Ecuador", "Nigeria"],
    "D": ["England", "Netherlands", "Senegal", "Costa Rica"],
    "E": ["France", "Belgium", "Croatia", "New Zealand"],
    "F": ["Portugal", "Turkey", "Czech Republic", "Egypt"],
    "G": ["Spain", "Germany", "Colombia", "Uruguay"],
    "H": ["Japan", "South Korea", "Saudi Arabia", "Paraguay"],
    "I": ["Canada", "Bosnia and Herzegovina", "Honduras", "Venezuela"],
    "J": ["Italy", "Serbia", "Australia", "Cameroon"],
    "K": ["Switzerland", "Poland", "Ukraine", "Tunisia"],
    "L": ["Denmark", "Iran", "Slovenia", "El Salvador"],
    "M": ["South Africa", "Bolivia", "Bahrain", "South Africa"],  # placeholder
    "N": ["Jordan", "Curacao", "Cabo Verde", "Tajikistan"],
    "O": ["Romania", "Mali", "Guatemala", "DR Congo"],
    "P": ["Norway", "Ghana", "Zambia", "Cuba"],
}

# FIFA ranking as of early 2026 (approximate — update before running)
FIFA_RANKINGS = {
    "Argentina": 1, "France": 2, "England": 3, "Spain": 4, "Belgium": 5,
    "Brazil": 6, "Portugal": 7, "Netherlands": 8, "Italy": 9, "Germany": 10,
    "Croatia": 11, "Morocco": 12, "Japan": 13, "USA": 14, "Mexico": 15,
    "Uruguay": 16, "Colombia": 17, "Senegal": 18, "Denmark": 19, "Switzerland": 20,
    "South Korea": 21, "Australia": 22, "Ecuador": 23, "Hungary": 24, "Turkey": 25,
    "Poland": 26, "Serbia": 27, "Ukraine": 28, "Chile": 29, "Peru": 30,
    "Iran": 31, "Norway": 32, "Czech Republic": 33, "Slovenia": 34, "Egypt": 35,
    "Saudi Arabia": 36, "Romania": 37, "South Africa": 38, "Canada": 39,
    "Bosnia and Herzegovina": 40, "Nigeria": 41, "Cameroon": 42, "Ghana": 43,
    "Tunisia": 44, "Algeria": 45, "Costa Rica": 46, "Panama": 47, "Bolivia": 48,
    "Paraguay": 49, "Honduras": 50, "Venezuela": 51, "New Zealand": 52,
    "Jordan": 53, "Uzbekistan": 54, "El Salvador": 55, "Guatemala": 56,
    "Cabo Verde": 57, "Curacao": 58, "Tajikistan": 60, "Zambia": 65,
    "Mali": 58, "DR Congo": 62, "Bahrain": 80, "Cuba": 100,
}


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


def compute_elo_ratings(df: pd.DataFrame, k: float = 20.0, initial_elo: float = 1500.0) -> pd.DataFrame:
    """
    Compute running Elo ratings for all teams across the dataset.
    Returns the original df with elo_home and elo_away columns added.
    """
    elo = {}

    def get_elo(team):
        return elo.get(team, initial_elo)

    def expected_score(ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    elo_home_list, elo_away_list = [], []

    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        eh, ea = get_elo(h), get_elo(a)

        elo_home_list.append(eh)
        elo_away_list.append(ea)

        # Determine result
        if row["home_score"] > row["away_score"]:
            sh, sa = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        # Update Elo
        elo[h] = eh + k * (sh - expected_score(eh, ea))
        elo[a] = ea + k * (sa - expected_score(ea, eh))

    df = df.copy()
    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # Save final Elo ratings
    elo_df = pd.DataFrame(list(elo.items()), columns=["team", "elo"]).sort_values("elo", ascending=False)
    elo_df.to_csv(PROCESSED / "current_elo.csv", index=False)
    print(f"[+] Computed Elo ratings for {len(elo)} teams")

    return df, elo


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for modelling.
    Each match produces TWO rows: one per team (home and away perspective).
    Target: goals scored by that team in the match.
    """
    rows = []

    for _, r in df.iterrows():
        # Home team row
        rows.append({
            "team": r["home_team"],
            "opponent": r["away_team"],
            "goals_scored": r["home_score"],
            "goals_conceded": r["away_score"],
            "is_neutral": r.get("neutral", False),
            "elo_diff": r["elo_diff"],           # positive = team has higher Elo
            "ranking_diff": r.get("ranking_diff", 0),
            "date": r["date"],
            "tournament": r.get("tournament", ""),
            "is_wc": "FIFA World Cup" in str(r.get("tournament", "")),
        })
        # Away team row
        rows.append({
            "team": r["away_team"],
            "opponent": r["home_team"],
            "goals_scored": r["away_score"],
            "goals_conceded": r["home_score"],
            "is_neutral": r.get("neutral", False),
            "elo_diff": -r["elo_diff"],
            "ranking_diff": -r.get("ranking_diff", 0),
            "date": r["date"],
            "tournament": r.get("tournament", ""),
            "is_wc": "FIFA World Cup" in str(r.get("tournament", "")),
        })

    features = pd.DataFrame(rows)
    features.to_csv(PROCESSED / "features.csv", index=False)
    print(f"[+] Built feature matrix: {len(features):,} rows")
    return features


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
        df, elo = compute_elo_ratings(df)
        features = build_features(df)
        fixtures = save_wc2026_fixtures()
        df.to_csv(PROCESSED / "matches_processed.csv", index=False)
        print("\n✓ Pipeline complete. Files saved to data/processed/")
