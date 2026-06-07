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
from tqdm import tqdm
from model import PoissonMatchPredictor
from data_loader import (
    TEAM_CONFEDERATION, CONFEDERATION_OFFSETS, HOST_NATIONS,
    WC2026_R32_BRACKET, WC2026_THIRD_PLACE_ELIGIBLE,
    WC2026_R16_PAIRS, WC2026_QF_PAIRS, WC2026_SF_PAIRS,
)

PROCESSED = Path("data/processed")


def get_team_elo(team: str, elo: dict, default: float = 1500.0) -> float:
    raw = elo.get(team, default)
    offset = CONFEDERATION_OFFSETS.get(TEAM_CONFEDERATION.get(team, "UEFA"), 0)
    return raw + offset


def get_team_ranking(team: str, rankings: dict) -> int | None:
    return rankings.get(team, None)


def get_team_form(team: str, form: dict, default: float = 1.3) -> float:
    val = form.get(team, default)
    return val if not np.isnan(val) else default


def get_team_squad(team: str, squad: dict, default: float = 74.0) -> float:
    return squad.get(team, default)


def get_team_rd(team: str, rd: dict, default: float = 100.0) -> float:
    return rd.get(team, default)


def assign_third_place(qualifying_thirds: list) -> dict:
    """
    Assign the 8 qualifying 3rd-place teams to their fixed R32 slots via backtracking.

    qualifying_thirds: list of (pts, gd, gf, team, group) sorted best-first.
    Returns {slot_id: team}.
    """
    group_to_team = {g: t for _, _, _, t, g in qualifying_thirds}
    qualifying_groups = set(group_to_team)
    slot_ids = list(WC2026_THIRD_PLACE_ELIGIBLE)
    assignment: dict = {}
    used: set = set()

    def backtrack(i: int) -> bool:
        if i == len(slot_ids):
            return True
        slot = slot_ids[i]
        candidates = (qualifying_groups & WC2026_THIRD_PLACE_ELIGIBLE[slot]) - used
        for g in sorted(candidates):
            assignment[slot] = group_to_team[g]
            used.add(g)
            if backtrack(i + 1):
                return True
            del assignment[slot]
            used.remove(g)
        return False

    if not backtrack(0):
        # Safety fallback — shouldn't happen with valid group combinations
        remaining = [group_to_team[g] for g in qualifying_groups if g not in used]
        for slot, team in zip([s for s in slot_ids if s not in assignment], remaining):
            assignment[slot] = team
    return assignment


def simulate_match(
    team_a: str,
    team_b: str,
    model: PoissonMatchPredictor,
    elo: dict,
    rankings: dict,
    neutral: bool = True,
) -> str:
    """
    Simulate a single match. Returns winner (or handles draw for group stage).
    For knockout: draws resolved by penalty shootout (50/50).
    """
    elo_a = get_team_elo(team_a, elo)
    elo_b = get_team_elo(team_b, elo)
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
    elo: dict,
    rankings: dict,
    form: dict,
    squad: dict,
    rd: dict,
) -> tuple:
    """
    Simulate a group stage. Returns (standings, points, gf, ga).
    Standings sorted by points → goal diff → goals for.
    """
    points = defaultdict(int)
    gf = defaultdict(int)
    ga = defaultdict(int)

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a, team_b = teams[i], teams[j]

            # Host nations play group games at home
            if team_a in HOST_NATIONS:
                home, away, is_neutral = team_a, team_b, False
            elif team_b in HOST_NATIONS:
                home, away, is_neutral = team_b, team_a, False
            else:
                home, away, is_neutral = team_a, team_b, True

            pred = model.predict_match(
                get_team_elo(home, elo),  get_team_elo(away, elo),
                get_team_ranking(home, rankings), get_team_ranking(away, rankings),
                is_neutral=is_neutral,
                form_home=get_team_form(home, form), form_away=get_team_form(away, form),
                squad_home=get_team_squad(home, squad), squad_away=get_team_squad(away, squad),
                rd_home=get_team_rd(home, rd), rd_away=get_team_rd(away, rd),
            )

            goals_home = np.random.poisson(pred["lambda_home"])
            goals_away = np.random.poisson(pred["lambda_away"])

            gf[home] += goals_home;  ga[home] += goals_away
            gf[away] += goals_away;  ga[away] += goals_home

            if goals_home > goals_away:
                points[home] += 3
            elif goals_home < goals_away:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

    standings = sorted(
        teams,
        key=lambda t: (points[t], gf[t] - ga[t], gf[t]),
        reverse=True,
    )
    return standings, dict(points), dict(gf), dict(ga)


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    model: PoissonMatchPredictor,
    elo: dict,
    rankings: dict,
    form: dict,
    squad: dict,
    rd: dict,
) -> str:
    """Simulate a knockout match. Draws go to penalties (50/50)."""
    elo_a   = get_team_elo(team_a, elo)
    elo_b   = get_team_elo(team_b, elo)
    rank_a  = get_team_ranking(team_a, rankings)
    rank_b  = get_team_ranking(team_b, rankings)
    form_a  = get_team_form(team_a, form)
    form_b  = get_team_form(team_b, form)
    squad_a = get_team_squad(team_a, squad)
    squad_b = get_team_squad(team_b, squad)
    rd_a    = get_team_rd(team_a, rd)
    rd_b    = get_team_rd(team_b, rd)

    pred = model.predict_match(elo_a, elo_b, rank_a, rank_b, is_neutral=True, form_home=form_a, form_away=form_b, squad_home=squad_a, squad_away=squad_b, rd_home=rd_a, rd_away=rd_b)

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
    elo: dict,
    rankings: dict,
    form: dict,
    squad: dict,
    rd: dict,
) -> tuple:
    """
    Simulate the full WC 2026 tournament using the official bracket structure.

    Returns:
      results:   team → stage where eliminated ("Group Stage" … "Winner")
      group_pos: team → group finish ("Group Winner" | "Group Runner-up" | "Group Stage")
    """
    results: dict = {}
    group_pos: dict = {}
    group_winners: dict = {}
    group_runners: dict = {}
    third_place: list = []

    # ── Group stage ───────────────────────────────────────────────────────────
    for gname, teams in groups.items():
        standings, pts, gf, ga = simulate_group(teams, model, elo, rankings, form, squad, rd)

        group_winners[gname] = standings[0]
        group_runners[gname] = standings[1]
        group_pos[standings[0]] = "Group Winner"
        group_pos[standings[1]] = "Group Runner-up"

        for team in standings[2:]:
            group_pos[team] = "Group Stage"
            results[team] = "Group Stage"

        t3 = standings[2]
        third_place.append((pts[t3], gf[t3] - ga[t3], gf[t3], t3, gname))

    # ── Best 8 third-place teams → assign to fixed R32 slots ─────────────────
    third_place.sort(reverse=True)
    slot_assignment = assign_third_place(third_place[:8])

    def resolve(spec):
        kind, ref = spec
        if kind == "W": return group_winners.get(ref)
        if kind == "R": return group_runners.get(ref)
        if kind == "T": return slot_assignment.get(ref)
        return None

    def sim_ko(a, b):
        if a is None: return b
        if b is None: return a
        return simulate_knockout_match(a, b, model, elo, rankings, form, squad, rd)

    def play_round(pairs, stage):
        winners = []
        for a, b in pairs:
            w = sim_ko(a, b)
            l = b if w == a else a
            results[w] = stage          # overwritten each round; final value = stage eliminated at
            if l is not None:
                results[l] = stage
            winners.append(w)
        return winners

    # ── Official WC 2026 bracket ──────────────────────────────────────────────
    r32_pairs = [(resolve(sa), resolve(sb)) for _, sa, sb in WC2026_R32_BRACKET]
    r32_w = play_round(r32_pairs, "Round of 32")

    r16_pairs = [(r32_w[i], r32_w[j]) for i, j in WC2026_R16_PAIRS]
    r16_w = play_round(r16_pairs, "Round of 16")

    qf_pairs = [(r16_w[i], r16_w[j]) for i, j in WC2026_QF_PAIRS]
    qf_w = play_round(qf_pairs, "Quarter-final")

    sf_pairs = [(qf_w[i], qf_w[j]) for i, j in WC2026_SF_PAIRS]
    sf_w = play_round(sf_pairs, "Semi-final")

    if len(sf_w) == 2 and sf_w[0] and sf_w[1]:
        winner = sim_ko(sf_w[0], sf_w[1])
        loser  = sf_w[1] if winner == sf_w[0] else sf_w[0]
        results[winner] = "Winner"
        results[loser]  = "Final"

    for teams in groups.values():
        for t in teams:
            if t not in results:
                results[t] = "Group Stage"

    return results, group_pos


def run_monte_carlo(
    groups: dict,
    model: PoissonMatchPredictor,
    elo: dict,
    rankings: dict,
    form: dict,
    squad: dict,
    rd: dict,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Run N simulations and compute stage probabilities for each team.

    Columns in output CSV:
      Group Winner / Group Runner-up — probability of finishing 1st / 2nd in group
      Group Stage  — probability of being eliminated before R32
      Round of 32 … Winner — probability of being eliminated at each knockout stage
    """
    knockout_stages = ["Group Stage", "Round of 32", "Round of 16",
                       "Quarter-final", "Semi-final", "Final", "Winner"]

    stage_counts = defaultdict(lambda: defaultdict(int))
    group_counts  = defaultdict(lambda: defaultdict(int))
    all_teams = [t for teams in groups.values() for t in teams]

    for _ in tqdm(range(n_simulations), desc="Simulating", unit="sim"):
        results, group_pos = simulate_tournament(groups, model, elo, rankings, form, squad, rd)
        for team, stage in results.items():
            stage_counts[team][stage] += 1
        for team, pos in group_pos.items():
            group_counts[team][pos] += 1

    rows = []
    for team in all_teams:
        row = {"team": team}
        row["Group Winner"]    = group_counts[team].get("Group Winner",    0) / n_simulations
        row["Group Runner-up"] = group_counts[team].get("Group Runner-up", 0) / n_simulations
        for s in knockout_stages:
            row[s] = stage_counts[team].get(s, 0) / n_simulations
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
        elo   = elo_df.set_index("team")["elo"].to_dict()
        rd    = elo_df.set_index("team")["rd"].to_dict() if "rd" in elo_df.columns else {}
        form  = pd.read_csv(PROCESSED / "current_form.csv").set_index("team")["form_scored"].to_dict()
        squad = pd.read_csv(PROCESSED / "current_squad_ratings.csv").set_index("team")["avg_overall"].to_dict()
        model = PoissonMatchPredictor.load()
        results = run_monte_carlo(WC2026_GROUPS, model, elo, FIFA_RANKINGS, form, squad, rd, n_simulations=10_000)
