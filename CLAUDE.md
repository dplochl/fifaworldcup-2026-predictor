# CLAUDE.md — Instructions for Claude Code

This file tells Claude Code how to work with this project.

## Project Overview

WC 2026 Match Predictor — Poisson regression model + Streamlit dashboard.

## Setup (run once)

```bash
pip install -r requirements.txt
# Download results.csv from Kaggle and place in data/raw/
python src/data_loader.py
python src/model.py
python src/simulator.py
streamlit run dashboard/app.py
```

## Key Files

- `src/data_loader.py` — data pipeline, Elo computation, feature engineering
- `src/model.py` — Poisson regression model (fit, predict, save/load)
- `src/simulator.py` — Monte Carlo tournament simulator
- `dashboard/app.py` — Streamlit dashboard (3 pages)

## Coding Style

- Type hints where helpful
- Docstrings on all functions
- Print progress with `[+]` prefix
- Save processed data to `data/processed/`

## Improvement Ideas (for Claude Code sessions)

1. **Better features**: Add head-to-head win rate, recent form (last 10 games), confederation
2. **FotMob ratings**: Scrape average player ratings per team as a feature
3. **Proper WC2026 bracket**: Wire up the actual R32 bracket pairings from the draw
4. **Calibration**: Plot predicted vs actual win rates to evaluate model
5. **UI upgrades**: Add flag emojis, animated bracket visualization
6. **Export**: Add "Export predictions as PDF" button
