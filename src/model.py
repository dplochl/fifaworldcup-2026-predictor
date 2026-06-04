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

    def _lambda(self, elo_diff: float, ranking_diff: float, is_neutral: bool, params: np.ndarray) -> float:
        """Compute expected goals (λ) given features and parameters."""
        b0, b_elo, b_rank, b_neutral = params
        return np.exp(b0 + b_elo * elo_diff + b_rank * ranking_diff + b_neutral * float(is_neutral))

    def _neg_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Negative log-likelihood for Poisson regression."""
        elo_diff, ranking_diff, is_neutral = X[:, 0], X[:, 1], X[:, 2]
        lam = np.exp(params[0] + params[1] * elo_diff + params[2] * ranking_diff + params[3] * is_neutral)
        lam = np.clip(lam, 1e-6, 20)
        nll = -np.sum(poisson.logpmf(y.astype(int), lam))
        return nll

    def fit(self, features: pd.DataFrame) -> "PoissonMatchPredictor":
        """Fit the model on historical match data."""
        df = features.dropna(subset=["elo_diff", "ranking_diff", "goals_scored"])

        X = df[["elo_diff", "ranking_diff", "is_neutral"]].values.astype(float)
        y = df["goals_scored"].values.astype(float)

        x0 = np.array([0.2, 0.001, -0.001, 0.0])
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )

        self.params = result.x
        self.is_fitted = True
        print(f"[+] Model fitted. Params: {dict(zip(['β₀','β_elo','β_rank','β_neutral'], self.params.round(4)))}")
        return self

    def predict_goals(self, elo_diff: float, ranking_diff: float, is_neutral: bool = True) -> float:
        """Predict expected goals for a team given match features."""
        assert self.is_fitted, "Model not fitted yet."
        return self._lambda(elo_diff, ranking_diff, is_neutral, self.params)

    def predict_match(
        self,
        elo_home: float,
        elo_away: float,
        rank_home: float = None,
        rank_away: float = None,
        is_neutral: bool = True,
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

        lam_home = self.predict_goals(elo_diff_home, rank_diff_home, is_neutral)
        lam_away = self.predict_goals(elo_diff_away, rank_diff_away, is_neutral)

        # Build score probability matrix
        mg = self.max_goals
        score_matrix = np.outer(
            poisson.pmf(np.arange(mg + 1), lam_home),
            poisson.pmf(np.arange(mg + 1), lam_away),
        )

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
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Model saved to {path}")

    @staticmethod
    def load(path: str = "models/poisson_model.pkl") -> "PoissonMatchPredictor":
        with open(path, "rb") as f:
            model = pickle.load(f)
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
    test = test.dropna(subset=["elo_diff", "ranking_diff"])
    preds = []
    for _, row in test.iterrows():
        lam = model.predict_goals(row["elo_diff"], row["ranking_diff"], row["is_neutral"])
        preds.append(lam)

    mae = np.mean(np.abs(test["goals_scored"].values - np.array(preds)))
    print(f"[+] Test MAE (goals): {mae:.4f}")

    return model


if __name__ == "__main__":
    print("=== WC 2026 Predictor — Model Training ===\n")
    model = train_and_evaluate()
