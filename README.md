# ⚽ WC 2026 Match Predictor

A machine learning model that predicts match outcomes for the 2026 FIFA World Cup using historical results, FIFA rankings, and Elo ratings.

## Overview

This project uses **Poisson Regression** to model the expected goals for each team in a match, then derives Win/Draw/Loss probabilities and simulates the entire tournament.

**Features used:**
- Elo rating differential between teams
- FIFA ranking differential
- Head-to-head historical record
- Confederation (proxy for playing style/level)
- Host nation advantage

## Project Structure

```
wc2026-predictor/
├── data/
│   ├── raw/               # Downloaded datasets (not committed)
│   └── processed/         # Cleaned, feature-engineered data
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory Data Analysis
├── src/
│   ├── data_loader.py     # Download & preprocess data
│   ├── features.py        # Feature engineering
│   ├── model.py           # Poisson regression model
│   └── simulator.py       # Tournament bracket simulator
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & process data
python src/data_loader.py

# 3. Train model
python src/model.py

# 4. Run dashboard
streamlit run dashboard/app.py
```

## Data Sources

- **Historical match results**: [International Football Results 1872–2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle)
- **Elo ratings**: [World Football Elo Ratings](http://www.eloratings.net/)
- **FIFA rankings**: [FIFA World Rankings](https://www.fifa.com/fifa-world-ranking)
- **WC 2026 fixtures**: [Wikipedia / official FIFA schedule](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup)

## Model

A **Poisson regression** model estimates λ (expected goals) for each team in a match:

```
λ_home = exp(β₀ + β₁·elo_diff + β₂·ranking_diff + β₃·is_neutral + ...)
```

Win/Draw/Loss probabilities are then derived from the joint Poisson distribution of both teams' scores.

## Results

| Stage | Top Picks |
|---|---|
| Winner | TBD after training |
| Final | TBD |
| Semifinals | TBD |

*(Updated after model training)*

## Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn, scipy
- Streamlit (dashboard)
- matplotlib / plotly (visualizations)

---

*Built during the 2026 FIFA World Cup as a learning project.*
