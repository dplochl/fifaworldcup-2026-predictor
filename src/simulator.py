"""
simulator.py
------------
Monte Carlo tournament simulator for the 2026 FIFA World Cup.

Simulates the full tournament N times using the Poisson model
to derive win probabilities for each team at each stage.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from model import PoissonMatchPredictor

PROCESSED = Path("data/processed")


def get_team_elo(team: str, elo_df: pd.DataFrame, default: float = 1500.0) -> float:
    row = elo_df[elo_df["team"] == team]
    return float(row["elo"].values[0]) if len(row) > 0 else default


def get_team_ranking(team: str, rankings: dict, default: int = 80) -> int:
    return rankings.get(team, default)


def simulate_match(
    team_a: str,
    team_b: str,
    model: PoissonMatchPredictor,
    elo_df: pd.DataFrame,
    rankings: dict,
    neutral: bool = True,
) -> str:
    """
    Simulate a single match. Returns winner (or handles draw for group stage).
    For knockout: draws resolved by penalty shootout (50/50).
    """
    elo_a = get_team_elo(team_a, elo_df)
    elo_b = get_team_elo(team_b, elo_df)
    rank_a = get_team_ranking(team_a, rankings)
    rank_b = get_team_ranking(team_b, rankings)

    pred = model.predict_match(elo_a, elo_b, rank_a, rank_b, neutral)

    # Sample outcome
    r = np.random.random()
    if r < pred["prob_home_win"]:
        return team_a, team_b  # winner, loser
    elif r < pred["prob_home_win"] + pred["prob_draw"]:
        return None, None  # draw (group stage)
    else:
        return team_b, team_a


def simulate_group(
    teams: list,
    model: PoissonMatchPredictor,
    elo_df: pd.DataFrame,
    rankings: dict,
) -> list:
    """
    Simulate a group stage. Returns list of teams sorted by points (top 2 advance).
    Simplified: no goal difference tiebreaker.
    """
    points = defaultdict(int)
    gf = defaultdict(int)
    ga = defaultdict(int)

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a, team_b = teams[i], teams[j]
            elo_a = get_team_elo(team_a, elo_df)
            elo_b = get_team_elo(team_b, elo_df)
            rank_a = get_team_ranking(team_a, rankings)
            rank_b = get_team_ranking(team_b, rankings)

            pred = model.predict_match(elo_a, elo_b, rank_a, rank_b, neutral=True)

            # Sample scoreline from Poisson
            goals_a = np.random.poisson(pred["lambda_home"])
            goals_b = np.random.poisson(pred["lambda_away"])

            gf[team_a] += goals_a
            ga[team_a] += goals_b
            gf[team_b] += goals_b
            ga[team_b] += goals_a

            if goals_a > goals_b:
                points[team_a] += 3
            elif goals_a < goals_b:
                points[team_b] += 3
            else:
                points[team_a] += 1
                points[team_b] += 1

    # Sort: points → goal diff → goals for
    standings = sorted(
        teams,
        key=lambda t: (points[t], gf[t] - ga[t], gf[t]),
        reverse=True,
    )
    return standings


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    model: PoissonMatchPredictor,
    elo_df: pd.DataFrame,
    rankings: dict,
) -> str:
    """Simulate a knockout match. Draws go to penalties (50/50)."""
    elo_a = get_team_elo(team_a, elo_df)
    elo_b = get_team_elo(team_b, elo_df)
    rank_a = get_team_ranking(team_a, rankings)
    rank_b = get_team_ranking(team_b, rankings)

    pred = model.predict_match(elo_a, elo_b, rank_a, rank_b, neutral=True)

    r = np.random.random()
    if r < pred["prob_home_win"]:
        return team_a
    elif r < pred["prob_home_win"] + pred["prob_draw"]:
        # Penalties: 50/50
        return team_a if np.random.random() < 0.5 else team_b
    else:
        return team_b


def simulate_tournament(
    groups: dict,
    model: PoissonMatchPredictor,
    elo_df: pd.DataFrame,
    rankings: dict,
) -> dict:
    """
    Simulate the full 2026 WC tournament (48 teams, 16 groups).
    Top 2 per group advance + 8 best 3rd-place teams = 32 teams in R32.

    Returns dict mapping each team to their furthest stage reached.
    """
    results = {}
    group_standings = {}

    # Group stage
    for group_name, teams in groups.items():
        standings = simulate_group(teams, model, elo_df, rankings)
        group_standings[group_name] = standings
        for i, team in enumerate(standings):
            if i == 0:
                results[team] = "Group Winner"
            elif i == 1:
                results[team] = "Group Runner-up"
            else:
                results[team] = "Group Stage"

    # Build R32 bracket (simplified pairing)
    group_names = list(group_standings.keys())
    r32_teams = []
    for gn in group_names:
        r32_teams.append(group_standings[gn][0])  # winner
        r32_teams.append(group_standings[gn][1])  # runner-up

    # Simulate knockout rounds
    stage_names = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]
    current_round = r32_teams

    for stage in stage_names:
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue
            winner = simulate_knockout_match(
                current_round[i], current_round[i + 1], model, elo_df, rankings
            )
            loser = current_round[i] if winner == current_round[i + 1] else current_round[i + 1]
            results[winner] = stage
            if stage != "Final":
                results[loser] = stage  # reached this stage but didn't advance
            next_round.append(winner)
        current_round = next_round

    if current_round:
        results[current_round[0]] = "Winner"

    return results


def run_monte_carlo(
    groups: dict,
    model: PoissonMatchPredictor,
    elo_df: pd.DataFrame,
    rankings: dict,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Run N simulations and compute win probabilities for each team at each stage.
    """
    stage_order = ["Group Stage", "Group Runner-up", "Group Winner",
                   "Round of 32", "Round of 16", "Quarter-final",
                   "Semi-final", "Final", "Winner"]

    stage_counts = defaultdict(lambda: defaultdict(int))
    all_teams = [t for teams in groups.values() for t in teams]

    for _ in range(n_simulations):
        result = simulate_tournament(groups, model, elo_df, rankings)
        for team, stage in result.items():
            stage_counts[team][stage] += 1

    # Build probability DataFrame
    rows = []
    for team in all_teams:
        row = {"team": team}
        for stage in stage_order:
            row[stage] = stage_counts[team].get(stage, 0) / n_simulations
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Winner", ascending=False)
    df.to_csv(PROCESSED / "simulation_results.csv", index=False)
    print(f"[+] Monte Carlo complete ({n_simulations:,} simulations)")
    print("\nTop 10 predicted winners:")
    print(df[["team", "Semi-final", "Final", "Winner"]].head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    from data_loader import WC2026_GROUPS, FIFA_RANKINGS

    print("=== WC 2026 — Tournament Simulator ===\n")

    elo_path = PROCESSED / "current_elo.csv"
    if not elo_path.exists():
        print("[!] Run data_loader.py first to generate Elo ratings.")
    else:
        elo_df = pd.read_csv(elo_path)
        model = PoissonMatchPredictor.load()
        results = run_monte_carlo(WC2026_GROUPS, model, elo_df, FIFA_RANKINGS, n_simulations=10_000)
