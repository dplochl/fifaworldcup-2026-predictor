"""
dashboard/app.py
----------------
Streamlit dashboard for the WC 2026 Match Predictor.

Run with: streamlit run dashboard/app.py
"""

import sys
sys.path.append("src")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

    .main { background-color: #0a0e1a; }
    h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
    .stMetric { background: #12192b; border-radius: 10px; padding: 12px; }

    .prob-card {
        background: linear-gradient(135deg, #12192b 0%, #1a2540 100%);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .prob-big { font-size: 2.5rem; font-weight: 700; color: #00d4ff; }
    .prob-label { font-size: 0.85rem; color: #8a9bc0; text-transform: uppercase; letter-spacing: 1px; }
    .win { color: #00ff88; }
    .draw { color: #ffcc00; }
    .loss { color: #ff4466; }
</style>
""", unsafe_allow_html=True)

PROCESSED = Path("data/processed")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_elo():
    p = PROCESSED / "current_elo.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_simulation_results():
    p = PROCESSED / "simulation_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_resource
def load_model():
    try:
        from model import PoissonMatchPredictor
        return PoissonMatchPredictor.load()
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ WC 2026 Predictor")
    st.markdown("*Poisson regression model*")
    st.divider()
    page = st.radio("Navigate", ["🏆 Tournament Odds", "⚔️ Match Predictor", "📊 Model Info"])

# ── Page: Tournament Odds ─────────────────────────────────────────────────────
if page == "🏆 Tournament Odds":
    st.title("2026 FIFA World Cup — Tournament Odds")
    st.caption("Based on 10,000 Monte Carlo simulations using Poisson regression on historical match data")

    sim = load_simulation_results()

    if sim is None:
        st.warning("⚠️ No simulation results found. Run `python src/simulator.py` first.")
        st.info("**Steps to get started:**\n1. Download `results.csv` from Kaggle\n2. Run `python src/data_loader.py`\n3. Run `python src/model.py`\n4. Run `python src/simulator.py`")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Win Probability — Top 20 Teams")
            top20 = sim.nlargest(20, "Winner")
            fig = go.Figure(go.Bar(
                x=(top20["Winner"] * 100).round(1),
                y=top20["team"],
                orientation="h",
                marker=dict(
                    color=(top20["Winner"] * 100),
                    colorscale="Teal",
                    showscale=False,
                ),
                text=[f"{v:.1f}%" for v in top20["Winner"] * 100],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=60, t=10, b=10),
                height=500,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Stage Probabilities")
            selected_team = st.selectbox("Select team", sim["team"].tolist())
            team_row = sim[sim["team"] == selected_team].iloc[0]

            stages = ["Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
            for stage in stages:
                val = team_row.get(stage, 0) * 100
                color = "#00ff88" if stage == "Winner" else "#00d4ff"
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e2a40'>
                    <span style='color:#8a9bc0'>{stage}</span>
                    <span style='color:{color}; font-weight:600'>{val:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Full Probability Table")
        display_cols = ["team", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
        display_df = sim[display_cols].copy()
        for col in display_cols[1:]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Page: Match Predictor ─────────────────────────────────────────────────────
elif page == "⚔️ Match Predictor":
    st.title("Head-to-Head Match Predictor")
    st.caption("Enter two teams to get predicted probabilities and expected scoreline")

    elo_df = load_elo()
    model = load_model()

    if elo_df is None or model is None:
        st.warning("⚠️ Run the data pipeline first (`data_loader.py` → `model.py`)")
    else:
        teams = sorted(elo_df["team"].tolist())

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            team_a = st.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
        with col2:
            st.markdown("<br><div style='text-align:center; font-size:1.5rem; color:#8a9bc0'>VS</div>", unsafe_allow_html=True)
        with col3:
            team_b = st.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)

        if st.button("⚽ Predict Match", type="primary"):
            from data_loader import FIFA_RANKINGS
            elo_a = float(elo_df[elo_df["team"] == team_a]["elo"].values[0]) if team_a in elo_df["team"].values else 1500.0
            elo_b = float(elo_df[elo_df["team"] == team_b]["elo"].values[0]) if team_b in elo_df["team"].values else 1500.0
            rank_a = FIFA_RANKINGS.get(team_a, 80)
            rank_b = FIFA_RANKINGS.get(team_b, 80)

            pred = model.predict_match(elo_a, elo_b, rank_a, rank_b, neutral=True)

            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='prob-card'>
                    <div class='prob-big win'>{pred['prob_home_win']*100:.1f}%</div>
                    <div class='prob-label'>{team_a} Win</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='prob-card'>
                    <div class='prob-big draw'>{pred['prob_draw']*100:.1f}%</div>
                    <div class='prob-label'>Draw</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='prob-card'>
                    <div class='prob-big loss'>{pred['prob_away_win']*100:.1f}%</div>
                    <div class='prob-label'>{team_b} Win</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <br>
            <div style='text-align:center; color:#8a9bc0; font-size:0.9rem'>
                Expected goals: <b style='color:white'>{pred['lambda_home']}</b> — <b style='color:white'>{pred['lambda_away']}</b>
                &nbsp;|&nbsp; Most likely score: <b style='color:#00d4ff'>{pred['most_likely_score']}</b>
            </div>
            """, unsafe_allow_html=True)

            # Score matrix heatmap
            st.divider()
            st.subheader("Score Probability Matrix")
            mg = min(8, model.max_goals)
            matrix = pred["score_matrix"][:mg+1, :mg+1]
            fig = px.imshow(
                (matrix * 100).round(2),
                labels=dict(x=f"{team_b} Goals", y=f"{team_a} Goals", color="%"),
                x=list(range(mg + 1)),
                y=list(range(mg + 1)),
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Model Info ──────────────────────────────────────────────────────────
elif page == "📊 Model Info":
    st.title("Model Documentation")

    st.markdown("""
    ## How It Works

    This predictor uses a **Poisson regression** model — the standard approach in football analytics for modelling goal-scoring.

    ### Core Idea

    For each team in a match, we model the number of goals scored as:

    > **Goals ~ Poisson(λ)**

    Where λ (expected goals) is estimated as:

    > **λ = exp(β₀ + β₁·elo_diff + β₂·ranking_diff + β₃·is_neutral)**

    The key insight: if we model both teams' goals independently, we can compute the full joint probability distribution of any scoreline — and derive Win/Draw/Loss probabilities from it.

    ### Features

    | Feature | Description |
    |---|---|
    | `elo_diff` | Difference in Elo ratings between the two teams |
    | `ranking_diff` | Difference in FIFA rankings |
    | `is_neutral` | Whether the match is played on neutral ground |

    ### Elo Ratings

    Elo ratings are computed iteratively from historical match data (all international matches since 2000), using a standard K=20 update rule.

    ### Data Sources

    - **Historical results**: Kaggle dataset (martj42), 1872–2024
    - **FIFA Rankings**: Approximate 2026 values (hardcoded)
    - **WC 2026 fixtures**: Hardcoded from official draw

    ### Limitations

    - Model doesn't account for squad injuries, suspensions, or form
    - FIFA rankings are approximate (update with current values for better accuracy)
    - Group stage draw bracket pairings in knockout rounds are simplified
    - FotMob/individual player ratings not yet integrated
    """)

    elo_df = load_elo()
    if elo_df is not None:
        st.divider()
        st.subheader("Current Elo Ratings (top 30)")
        st.dataframe(elo_df.head(30), use_container_width=True, hide_index=True)
