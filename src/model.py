"""
model.py
--------
Poisson regression model for predicting match outcomes.

Core idea:
  - Model goals scored by each team as Poisson(λ)
  - λ = exp(intercept + β·elo_diff + γ·ranking_diff + ...)
  - Win/Draw/Loss probabilities from the joint distribution
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from pathlib import Path
import pickle

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


class PoissonMatchPredictor:
    """
    Poisson regression model for international football match prediction.

    For each match, models:
      λ_team = exp(β₀ + β₁·elo_diff + β₂·ranking_diff + β₃·is_neutral)

    Win/Draw/Loss probs derived from P(goals_home = i, goals_away = j).
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.params = None
        self.is_fitted = False

    def _lambda(self, elo_diff: float, ranking_diff: float, is_home: int, form_scored: float, squad_rating_diff: float, params: np.ndarray) -> float:
        """Compute expected goals (λ) given features and parameters."""
        b0, b_elo, b_rank, b_home, b_form, b_squad = params
        return np.exp(
            b0 + b_elo * elo_diff + b_rank * ranking_diff
            + b_home * is_home + b_form * form_scored
            + b_squad * squad_rating_diff
        )

    def _neg_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Weighted negative log-likelihood for Poisson regression."""
        elo_diff, ranking_diff, is_home, form_scored, squad_diff = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        log_lam = (params[0] + params[1] * elo_diff + params[2] * ranking_diff
                   + params[3] * is_home + params[4] * form_scored + params[5] * squad_diff)
        lam = np.exp(np.clip(log_lam, -10, 3))  # clip before exp to prevent overflow
        nll = -np.sum(weights * poisson.logpmf(y.astype(int), lam))
        return nll

    def fit(self, features: pd.DataFrame) -> "PoissonMatchPredictor":
        """Fit the model on historical match data."""
        df = features.dropna(subset=["elo_diff", "ranking_diff", "goals_scored", "is_home", "form_scored"])
        df["squad_rating_diff"] = df["squad_rating_diff"].fillna(0) if "squad_rating_diff" in df.columns else 0

        X = df[["elo_diff", "ranking_diff", "is_home", "form_scored", "squad_rating_diff"]].values.astype(float)
        y = df["goals_scored"].values.astype(float)
        weights = df["match_weight"].values.astype(float) if "match_weight" in df.columns else np.ones(len(df))

        x0 = np.array([0.0, 0.001, -0.001, 0.1, 0.1, 0.01])
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(X, y, weights),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )

        self.params = result.x
        self.is_fitted = True
        print(f"[+] Model fitted. Params: {dict(zip(['β₀','β_elo','β_rank','β_home','β_form','β_squad'], self.params.round(4)))}")
        return self

    def predict_goals(self, elo_diff: float, ranking_diff: float, is_home: int = 0, form_scored: float = 1.3, squad_rating_diff: float = 0.0, rd_combined: float = 100.0) -> float:
        """Predict expected goals for a team given match features."""
        assert self.is_fitted, "Model not fitted yet."
        return self._lambda(elo_diff, ranking_diff, is_home, form_scored, squad_rating_diff, self.params)

    def predict_match(
        self,
        elo_home: float,
        elo_away: float,
        rank_home: float = None,
        rank_away: float = None,
        is_neutral: bool = True,
        form_home: float = 1.3,
        form_away: float = 1.3,
        squad_home: float = 74.0,
        squad_away: float = 74.0,
        rd_home: float = 100.0,
        rd_away: float = 100.0,
    ) -> dict:
        """
        Predict Win/Draw/Loss probabilities for a match.

        Returns dict with:
          - lambda_home, lambda_away: expected goals
          - prob_home_win, prob_draw, prob_away_win
          - score_matrix: (max_goals x max_goals) probability matrix
        """
        assert self.is_fitted

        elo_diff_home = elo_home - elo_away
        elo_diff_away = elo_away - elo_home
        rank_diff_home = (rank_home - rank_away) if rank_home and rank_away else 0
        rank_diff_away = -rank_diff_home
        home_advantage = 0 if is_neutral else 1
        squad_diff = squad_home - squad_away

        lam_home = self.predict_goals(elo_diff_home, rank_diff_home, home_advantage, form_home,  squad_diff)
        lam_away = self.predict_goals(elo_diff_away, rank_diff_away, 0,              form_away, -squad_diff)

        # Build score probability matrix with Dixon-Coles correction.
        # The basic Poisson model underestimates 0-0 and 1-1 draws and
        # overestimates 0-1 / 1-0. The DC correction factor ρ (rho) fixes
        # this by adjusting probabilities for the four low-scoring outcomes.
        # ρ ≈ -0.13 is the standard estimate for international football
        # (Dixon & Coles, 1997).
        mg = self.max_goals
        rho = -0.13
        score_matrix = np.outer(
            poisson.pmf(np.arange(mg + 1), lam_home),
            poisson.pmf(np.arange(mg + 1), lam_away),
        )
        # Apply correction only to the 2×2 low-score block
        score_matrix[0, 0] *= (1 - rho * lam_home * lam_away)
        score_matrix[0, 1] *= (1 + rho * lam_home)
        score_matrix[1, 0] *= (1 + rho * lam_away)
        score_matrix[1, 1] *= (1 - rho)
        score_matrix = np.clip(score_matrix, 0, None)
        score_matrix /= score_matrix.sum()  # renormalise

        prob_home_win = np.tril(score_matrix, -1).sum()
        prob_draw = np.trace(score_matrix)
        prob_away_win = np.triu(score_matrix, 1).sum()

        # Most likely scoreline
        flat_idx = np.argmax(score_matrix)
        most_likely_home = flat_idx // (mg + 1)
        most_likely_away = flat_idx % (mg + 1)

        return {
            "lambda_home": round(lam_home, 3),
            "lambda_away": round(lam_away, 3),
            "prob_home_win": round(prob_home_win, 4),
            "prob_draw": round(prob_draw, 4),
            "prob_away_win": round(prob_away_win, 4),
            "most_likely_score": f"{most_likely_home}–{most_likely_away}",
            "score_matrix": score_matrix,
        }

    def save(self, path: str = "models/poisson_model.pkl"):
        data = {
            "params": self.params.tolist(),
            "max_goals": self.max_goals,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[+] Model saved to {path}")

    @staticmethod
    def load(path: str = "models/poisson_model.pkl") -> "PoissonMatchPredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = PoissonMatchPredictor(max_goals=data["max_goals"])
        model.params = np.array(data["params"])
        model.is_fitted = data["is_fitted"]
        print(f"[+] Model loaded from {path}")
        return model


def train_and_evaluate():
    """Train model on historical data and print evaluation metrics."""
    features_path = PROCESSED / "features.csv"
    if not features_path.exists():
        print("[!] Run data_loader.py first to generate features.csv")
        return

    df = pd.read_csv(features_path)

    # Train/test split by time
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp("2022-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    print(f"[+] Train: {len(train):,} rows | Test: {len(test):,} rows")

    model = PoissonMatchPredictor()
    model.fit(train)
    model.save()

    # Simple evaluation: MAE on goals
    test = test.dropna(subset=["elo_diff", "ranking_diff", "is_home", "form_scored", "goals_scored"])
    test["squad_rating_diff"] = test["squad_rating_diff"].fillna(0) if "squad_rating_diff" in test.columns else 0
    preds = []
    for _, row in test.iterrows():
        lam = model.predict_goals(row["elo_diff"], row["ranking_diff"], row["is_home"], row["form_scored"], row.get("squad_rating_diff", 0))
        preds.append(lam)

    mae = np.mean(np.abs(test["goals_scored"].values - np.array(preds)))
    print(f"[+] Test MAE (goals): {mae:.4f}")

    # Match-level evaluation: log-loss and F1
    from sklearn.metrics import log_loss, f1_score

    # One row per match: take team < opponent alphabetically as "team A"
    test["date_str"] = test["date"].astype(str)
    test_a = test[test["team"].astype(str) < test["opponent"].astype(str)].copy()
    test_b = test[test["team"].astype(str) > test["opponent"].astype(str)][
        ["team", "opponent", "date_str", "form_scored"]
    ].rename(columns={"team": "opponent", "opponent": "team", "form_scored": "form_opponent"})

    test_matches = test_a.merge(test_b, on=["team", "opponent", "date_str"], how="left")
    test_matches["form_opponent"] = test_matches["form_opponent"].fillna(1.3)
    test_matches["squad_rating_diff"] = test_matches["squad_rating_diff"].fillna(0)
    test_matches["ranking_diff"] = test_matches["ranking_diff"].fillna(0)

    y_true = np.where(
        test_matches["goals_scored"] > test_matches["goals_conceded"], 0,
        np.where(test_matches["goals_scored"] == test_matches["goals_conceded"], 1, 2)
    )

    probs = []
    for _, row in test_matches.iterrows():
        ed = row["elo_diff"]
        rd = row["ranking_diff"]
        ih = row["is_home"]
        fa = row["form_scored"]
        fb = row["form_opponent"]
        sq = row["squad_rating_diff"]

        lam_a = model.predict_goals(ed, rd, ih, fa, sq)
        lam_b = model.predict_goals(-ed, -rd, 0, fb, -sq)

        mg = model.max_goals
        sm = np.outer(poisson.pmf(np.arange(mg + 1), lam_a),
                      poisson.pmf(np.arange(mg + 1), lam_b))
        p = [np.tril(sm, -1).sum(), np.trace(sm), np.triu(sm, 1).sum()]
        p = [max(x, 1e-7) for x in p]
        s = sum(p)
        probs.append([x / s for x in p])

    probs = np.array(probs)
    y_pred = np.argmax(probs, axis=1)

    ll = log_loss(y_true, probs)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"[+] Log-loss (W/D/L): {ll:.4f}")
    print(f"[+] F1 score (weighted): {f1:.4f}")

    return model


if __name__ == "__main__":
    print("=== WC 2026 Predictor — Model Training ===\n")
    model = train_and_evaluate()
