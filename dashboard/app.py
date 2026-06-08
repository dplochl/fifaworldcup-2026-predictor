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

    /* ── Base ── */
    .stApp, .main { background-color: #070C1A; }

    h1 {
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 3px;
        background: linear-gradient(90deg, #C4981F 0%, #B55A28 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; color: #E8E8E8; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060D20 0%, #0A1628 100%);
        border-right: 1px solid #1E2F50;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0E1628 0%, #121E38 100%);
        border: 1px solid #1E2F50;
        border-left: 3px solid #C4981F;
        border-radius: 10px;
        padding: 14px;
    }

    /* ── Prob cards ── */
    .prob-card {
        background: linear-gradient(135deg, #0E1628 0%, #121E38 100%);
        border: 1px solid #1E2F50;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .prob-big { font-size: 2.8rem; font-weight: 700; }
    .prob-label { font-size: 0.85rem; color: #8A9BC0; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }
    .win  { color: #3EA370; }
    .draw { color: #C4A030; }
    .loss { color: #B84248; }

    /* ── Stage rows ── */
    .stage-row {
        display: flex; justify-content: space-between;
        padding: 8px 12px;
        border-bottom: 1px solid #1E2F50;
        border-radius: 6px;
        margin: 2px 0;
    }
    .stage-row:hover { background: rgba(255,182,39,0.05); }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #C4981F, #B55A28) !important;
        color: #070C1A !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        letter-spacing: 1px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        color: #8A9BC0;
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 1.5px;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        color: #C4981F !important;
        border-bottom-color: #C4981F !important;
    }

    /* ── Divider ── */
    hr { border-color: #1E2F50 !important; }
</style>
""", unsafe_allow_html=True)

PROCESSED = Path("data/processed")

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_elo():
    p = PROCESSED / "current_elo.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_form():
    p = PROCESSED / "current_form.csv"
    return pd.read_csv(p).set_index("team")["form_scored"].to_dict() if p.exists() else {}

@st.cache_data
def load_squad():
    p = PROCESSED / "current_squad_ratings.csv"
    return pd.read_csv(p).set_index("team")["avg_overall"].to_dict() if p.exists() else {}

@st.cache_data
def load_rd():
    p = PROCESSED / "current_elo.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "rd" in df.columns:
            return df.set_index("team")["rd"].to_dict()
    return {}

@st.cache_data
def load_simulation_results():
    p = PROCESSED / "simulation_results.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_model():
    try:
        from model import PoissonMatchPredictor
        return PoissonMatchPredictor.load()
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# ── Shared helpers ────────────────────────────────────────────────────────────
def _adj_elo(team, elo_dict):
    from data_loader import TEAM_CONFEDERATION, CONFEDERATION_OFFSETS
    raw    = elo_dict.get(team, 1500.0)
    offset = CONFEDERATION_OFFSETS.get(TEAM_CONFEDERATION.get(team, "UEFA"), 0)
    return raw + offset

def predict(team_a, team_b, elo_dict, form, squad, rd_dict, neutral=True):
    """Run predict_match with all preloaded data."""
    model = load_model()
    if model is None:
        return None
    from data_loader import FIFA_RANKINGS
    return model.predict_match(
        _adj_elo(team_a, elo_dict), _adj_elo(team_b, elo_dict),
        FIFA_RANKINGS.get(team_a), FIFA_RANKINGS.get(team_b),
        is_neutral=neutral,
        form_home=form.get(team_a, 1.3), form_away=form.get(team_b, 1.3),
        squad_home=squad.get(team_a, 74.0), squad_away=squad.get(team_b, 74.0),
        rd_home=rd_dict.get(team_a, 100.0), rd_away=rd_dict.get(team_b, 100.0),
    )

def prob_row(label, value, color="#C4981F"):
    st.markdown(f"""
    <div class='stage-row'>
        <span style='color:#8A9BC0'>{label}</span>
        <span style='color:{color}; font-weight:600'>{value:.1f}%</span>
    </div>""", unsafe_allow_html=True)

def reach_prob(row, stage: str) -> float:
    """P of REACHING a stage = sum of P(eliminated at this stage or later)."""
    _order = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
    if stage not in _order:
        return 0.0
    return float(sum(row.get(s, 0) for s in _order[_order.index(stage):]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px;border-bottom:1px solid #1E2F50;margin-bottom:16px'>
        <div style='font-size:2.6rem'>⚽</div>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:2rem;letter-spacing:3px;
                    background:linear-gradient(90deg,#C4981F,#B55A28);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;line-height:1.1'>
            FIFA WORLD CUP
        </div>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:1.1rem;letter-spacing:4px;
                    color:#8A9BC0;margin-top:2px'>2026 PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("*Poisson regression + Monte Carlo*")
    page = st.radio("", [
        "🏆 Tournament Odds",
        "🔍 Model vs. FIFA Rankings",
        "📊 Group Tables",
        "⚔️ Match Predictor",
        "📖 Model Info",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# WC 2026 — Page 1: Tournament Odds
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Tournament Odds":
    st.title("2026 FIFA World Cup — Tournament Odds")
    st.caption("Based on 10,000 Monte Carlo simulations using Poisson regression on historical match data")

    sim = load_simulation_results()

    if sim is None:
        st.warning("⚠️ No simulation results found. Run `python src/simulator.py` first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Win Probability — Top 10 Teams")
            top20 = sim.nlargest(10, "Winner")
            fig = go.Figure(go.Bar(
                x=(top20["Winner"] * 100).round(1),
                y=top20["team"],
                orientation="h",
                marker=dict(color=(top20["Winner"] * 100), colorscale=[[0, "#1E2F50"], [0.5, "#B55A28"], [1, "#C4981F"]], showscale=False),
                text=[f"{v:.1f}%" for v in top20["Winner"] * 100],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), margin=dict(l=10, r=60, t=10, b=10),
                height=500,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Stage Probabilities")
            st.caption("Probability of *reaching* each stage")
            selected_team = st.selectbox("Select team", sim["team"].tolist())
            team_row = sim[sim["team"] == selected_team].iloc[0]
            for stage in ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]:
                val   = reach_prob(team_row, stage) * 100
                color = "#3EA370" if stage == "Winner" else "#C4981F"
                prob_row(stage, val, color)

        st.divider()
        st.subheader("Full Probability Table")
        from data_loader import TEAM_FLAGS
        display_cols = ["team", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
        display_df = sim[display_cols].copy()
        display_df["team"] = display_df["team"].map(lambda t: f"{TEAM_FLAGS.get(t, '')} {t}")
        for col in display_cols[1:]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# WC 2026 — Page 3: Model vs. FIFA Rankings
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Model vs. FIFA Rankings":
    st.title("🔍 Model vs. FIFA Rankings")
    st.caption("Which teams does the model rate differently from FIFA?")

    from data_loader import WC2026_GROUPS, FIFA_RANKINGS, TEAM_FLAGS

    sim = load_simulation_results()
    if sim is None:
        st.warning("⚠️ Run `python src/simulator.py` first.")
        st.stop()

    # All WC teams
    wc_teams = [t for teams in WC2026_GROUPS.values() for t in teams]

    # FIFA rank (lower = better)
    fifa_ranks = {t: FIFA_RANKINGS.get(t, 999) for t in wc_teams}

    # Model rank (higher win prob = better)
    model_win = {row["team"]: row.get("Winner", 0) for _, row in sim.iterrows()}
    model_ranks = {
        t: rank + 1
        for rank, (t, _) in enumerate(
            sorted([(t, model_win.get(t, 0)) for t in wc_teams], key=lambda x: x[1], reverse=True)
        )
    }
    fifa_rank_sorted = {
        t: rank + 1
        for rank, (t, _) in enumerate(
            sorted([(t, fifa_ranks[t]) for t in wc_teams], key=lambda x: x[1])
        )
    }

    df_cmp = pd.DataFrame({
        "team":       wc_teams,
        "fifa_rank":  [fifa_rank_sorted[t] for t in wc_teams],
        "model_rank": [model_ranks[t]      for t in wc_teams],
    })
    df_cmp["rank_diff"] = df_cmp["fifa_rank"] - df_cmp["model_rank"]
    df_cmp = df_cmp.sort_values("rank_diff")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Overrated by FIFA")
        st.caption("FIFA ranks these teams much higher than the model does")
        over = df_cmp.nsmallest(5, "rank_diff")
        for _, row in over.iterrows():
            flag = TEAM_FLAGS.get(row["team"], "")
            st.markdown(
                f"**{flag} {row['team']}** — FIFA #{row['fifa_rank']} → Model #{row['model_rank']} "
                f"<span style='color:#B84248'>({row['rank_diff']:+.0f})</span>",
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("🟢 Underrated by FIFA")
        st.caption("The model ranks these teams much higher than FIFA does")
        under = df_cmp.nlargest(5, "rank_diff")
        for _, row in under.iterrows():
            flag = TEAM_FLAGS.get(row["team"], "")
            st.markdown(
                f"**{flag} {row['team']}** — FIFA #{row['fifa_rank']} → Model #{row['model_rank']} "
                f"<span style='color:#3EA370'>({row['rank_diff']:+.0f})</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("All 48 Teams — Ranking Divergence")

    df_chart = df_cmp.sort_values("rank_diff")
    df_chart["flag_team"] = df_chart["team"].map(lambda t: f"{TEAM_FLAGS.get(t,'')} {t}")
    df_chart["color"]     = df_chart["rank_diff"].map(lambda d: "#3EA370" if d > 0 else "#B84248")

    fig = go.Figure(go.Bar(
        x=df_chart["rank_diff"],
        y=df_chart["flag_team"],
        orientation="h",
        marker_color=df_chart["color"].tolist(),
        text=df_chart["rank_diff"].map(lambda d: f"{d:+.0f}"),
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=900,
        margin=dict(l=10, r=50, t=20, b=10),
        xaxis=dict(title="FIFA rank − Model rank  (positive = model rates higher)", showgrid=False, zeroline=True,
                   zerolinecolor="#1E2F50"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# WC 2026 — Page 4: Group Tables
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Group Tables":
    from data_loader import WC2026_GROUPS, TEAM_FLAGS, HOST_NATIONS
    from itertools import combinations

    st.title("⚽ Group Stage Predictor")
    st.caption("Model's most likely outcome for every group match")

    elo_df = load_elo()
    model  = load_model()
    if elo_df is None or model is None:
        st.warning("⚠️ Run the data pipeline first.")
        st.stop()

    elo_dict = elo_df.set_index("team")["elo"].to_dict()
    form     = load_form()
    squad    = load_squad()
    rd_dict  = load_rd()

    # ── helpers ───────────────────────────────────────────────────────────────
    def flag_t(t):
        return f"{TEAM_FLAGS.get(t, '')} {t}"

    def predict_game(t1, t2):
        """Predict a group match; handles host nation home advantage."""
        if t2 in HOST_NATIONS and t1 not in HOST_NATIONS:
            p = predict(t2, t1, elo_dict, form, squad, rd_dict, neutral=False)
            if not p:
                return None
            hg, ag = map(int, p["most_likely_score"].split("–"))
            return dict(t1=t1, t2=t2, g1=ag, g2=hg,
                        p1=p["prob_away_win"], pd=p["prob_draw"], p2=p["prob_home_win"])
        neutral = t1 not in HOST_NATIONS
        p = predict(t1, t2, elo_dict, form, squad, rd_dict, neutral=neutral)
        if not p:
            return None
        hg, ag = map(int, p["most_likely_score"].split("–"))
        return dict(t1=t1, t2=t2, g1=hg, g2=ag,
                    p1=p["prob_home_win"], pd=p["prob_draw"], p2=p["prob_away_win"])

    def match_card(m):
        """Render a group match as an HTML card."""
        g1, g2 = m["g1"], m["g2"]
        w = m["t1"] if g1 > g2 else (m["t2"] if g2 > g1 else None)
        t1_style = "font-weight:700;color:#C4981F" if w == m["t1"] else "color:#E8E8E8"
        t2_style = "font-weight:700;color:#C4981F" if w == m["t2"] else "color:#E8E8E8"
        prob_row = (f"<div style='font-size:0.68rem;color:#8A9BC0;"
                    f"text-align:center;margin-top:3px'>"
                    f"{m['p1']*100:.0f}% · {m['pd']*100:.0f}% · {m['p2']*100:.0f}%</div>")
        return (f"<div style='background:linear-gradient(135deg,#0E1628,#121E38);"
                f"border:1px solid #1E2F50;border-radius:8px;padding:10px 14px;margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:6px'>"
                f"<span style='{t1_style};flex:1'>{flag_t(m['t1'])}</span>"
                f"<span style='color:#C4A030;font-weight:700;font-size:1rem;"
                f"white-space:nowrap'>{g1}–{g2}</span>"
                f"<span style='{t2_style};flex:1;text-align:right'>{flag_t(m['t2'])}</span>"
                f"</div>{prob_row}</div>")

    def compute_standings(matches, teams):
        tbl = {t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0) for t in teams}
        for m in matches:
            t1, t2, g1, g2 = m["t1"], m["t2"], m["g1"], m["g2"]
            tbl[t1]["P"] += 1;  tbl[t1]["GF"] += g1;  tbl[t1]["GA"] += g2
            tbl[t2]["P"] += 1;  tbl[t2]["GF"] += g2;  tbl[t2]["GA"] += g1
            if g1 > g2:
                tbl[t1]["W"] += 1;  tbl[t1]["Pts"] += 3;  tbl[t2]["L"] += 1
            elif g1 == g2:
                tbl[t1]["D"] += 1;  tbl[t1]["Pts"] += 1
                tbl[t2]["D"] += 1;  tbl[t2]["Pts"] += 1
            else:
                tbl[t2]["W"] += 1;  tbl[t2]["Pts"] += 3;  tbl[t1]["L"] += 1
        for t in tbl:
            tbl[t]["GD"] = tbl[t]["GF"] - tbl[t]["GA"]
        ranked = sorted(teams,
                        key=lambda t: (tbl[t]["Pts"], tbl[t]["GD"], tbl[t]["GF"]),
                        reverse=True)
        return ranked, tbl

    # ── compute predictions (cached in session state per elo snapshot) ────────
    cache_key = f"groups_{round(elo_df['elo'].sum())}"
    if cache_key not in st.session_state:
        with st.spinner("Running group stage predictions…"):
            gp, standings, thirds = {}, {}, []
            for group, teams in WC2026_GROUPS.items():
                matches = [m for t1, t2 in combinations(teams, 2)
                           if (m := predict_game(t1, t2)) is not None]
                gp[group] = matches
                ranked, tbl = compute_standings(matches, teams)
                standings[group] = {"ranked": ranked, "table": tbl}
                t3 = ranked[2]
                thirds.append(dict(team=t3, group=group,
                                   pts=tbl[t3]["Pts"], gd=tbl[t3]["GD"], gf=tbl[t3]["GF"]))
            thirds.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
            st.session_state[cache_key] = dict(gp=gp, standings=standings, thirds=thirds)

    data      = st.session_state[cache_key]
    gp        = data["gp"]
    standings = data["standings"]
    thirds    = data["thirds"]

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["⚽ Group Matches", "📊 Group Standings"])

    # ── Tab 1: Group Matches ──────────────────────────────────────────────────
    with tab1:
        groups = list(WC2026_GROUPS.keys())
        for row_start in range(0, 12, 3):
            cols = st.columns(3)
            for col, group in zip(cols, groups[row_start:row_start + 3]):
                with col:
                    st.markdown(
                        f"<div style='font-family:\"Bebas Neue\",sans-serif;font-size:1.1rem;"
                        f"letter-spacing:2px;color:#C4981F;margin-bottom:4px'>GROUP {group}</div>",
                        unsafe_allow_html=True)
                    for m in gp[group]:
                        st.markdown(match_card(m), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab 2: Group Standings ────────────────────────────────────────────────
    with tab2:
        best_third_groups = {t["group"] for t in thirds[:8]}
        st.caption("🟢 Qualifies automatically · 🟡 Best 3rd-place qualifier")

        groups = list(WC2026_GROUPS.keys())
        for row_start in range(0, 12, 3):
            cols = st.columns(3)
            for col, group in zip(cols, groups[row_start:row_start + 3]):
                with col:
                    st.markdown(
                        f"<div style='font-family:\"Bebas Neue\",sans-serif;font-size:1.1rem;"
                        f"letter-spacing:2px;color:#C4981F;margin-bottom:4px'>GROUP {group}</div>",
                        unsafe_allow_html=True)
                    ranked = standings[group]["ranked"]
                    tbl    = standings[group]["table"]
                    medals = ["🥇", "🥈", "🥉", "4️⃣"]
                    for i, t in enumerate(ranked):
                        td = tbl[t]
                        if i < 2:
                            color = "#3EA370"
                        elif i == 2 and group in best_third_groups:
                            color = "#C4A030"
                        else:
                            color = "#8A9BC0"
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:5px 2px;border-bottom:1px solid #1E2F50;font-size:0.85rem'>"
                            f"<span>{medals[i]} <span style='color:{color}'>{flag_t(t)}</span></span>"
                            f"<span style='color:#8A9BC0'>{td['W']}W {td['D']}D {td['L']}L &nbsp;"
                            f"{td['GF']}:{td['GA']} &nbsp; <b style='color:{color}'>"
                            f"{td['Pts']}pts</b></span></div>",
                            unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Best 8 Third-Place Qualifiers")
        t3_rows = [{"#": i + 1, "Team": flag_t(t["team"]), "Group": t["group"],
                    "Pts": t["pts"], "GD": f"{t['gd']:+d}", "GF": t["gf"]}
                   for i, t in enumerate(thirds[:8])]
        st.dataframe(pd.DataFrame(t3_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# WC 2026 — Page 4: Match Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Match Predictor":
    st.title("Head-to-Head Match Predictor")
    st.caption("Enter two teams to get predicted probabilities and expected scoreline")

    elo_df = load_elo()
    model  = load_model()

    if elo_df is None or model is None:
        st.warning("⚠️ Run the data pipeline first (`data_loader.py` → `model.py`)")
    else:
        from data_loader import FIFA_RANKINGS, TEAM_FLAGS, WC2026_GROUPS
        wc_teams      = {t for teams in WC2026_GROUPS.values() for t in teams}
        teams         = sorted([t for t in elo_df["team"].tolist() if t in wc_teams])
        teams_display = [f"{TEAM_FLAGS.get(t, '')} {t}" for t in teams]
        team_map      = {d: t for d, t in zip(teams_display, teams)}

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            default_a    = teams_display[teams.index("Brazil")] if "Brazil" in teams else teams_display[0]
            team_a_disp  = st.selectbox("Team A", teams_display, index=teams_display.index(default_a))
            team_a       = team_map[team_a_disp]
        with col2:
            st.markdown("<br><div style='text-align:center;font-size:1.5rem;color:#8A9BC0'>VS</div>",
                        unsafe_allow_html=True)
        with col3:
            default_b    = teams_display[teams.index("France")] if "France" in teams else teams_display[1]
            team_b_disp  = st.selectbox("Team B", teams_display, index=teams_display.index(default_b))
            team_b       = team_map[team_b_disp]

        if st.button("⚽ Predict Match", type="primary"):
            elo_dict = elo_df.set_index("team")["elo"].to_dict()
            form     = load_form()
            squad    = load_squad()
            rd_dict  = load_rd()
            pred     = predict(team_a, team_b, elo_dict, form, squad, rd_dict, neutral=True)

            if pred:
                st.divider()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class='prob-card'>
                        <div class='prob-big win'>{pred['prob_home_win']*100:.1f}%</div>
                        <div class='prob-label'>{team_a} Win</div></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class='prob-card'>
                        <div class='prob-big draw'>{pred['prob_draw']*100:.1f}%</div>
                        <div class='prob-label'>Draw</div></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class='prob-card'>
                        <div class='prob-big loss'>{pred['prob_away_win']*100:.1f}%</div>
                        <div class='prob-label'>{team_b} Win</div></div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <br><div style='text-align:center;color:#8A9BC0;font-size:0.9rem'>
                    Expected goals: <b style='color:white'>{pred['lambda_home']}</b> —
                    <b style='color:white'>{pred['lambda_away']}</b>
                    &nbsp;|&nbsp; Most likely score: <b style='color:#C4981F'>{pred['most_likely_score']}</b>
                </div>""", unsafe_allow_html=True)

                st.divider()
                st.subheader("Score Probability Matrix")
                mg     = min(8, model.max_goals)
                matrix = pred["score_matrix"][:mg+1, :mg+1]
                fig    = px.imshow(
                    (matrix * 100).round(2),
                    labels=dict(x=f"{team_b} Goals", y=f"{team_a} Goals", color="%"),
                    x=list(range(mg + 1)), y=list(range(mg + 1)),
                    color_continuous_scale="Blues", aspect="auto",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# WC 2026 — Page 5: Model Info
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Model Info":
    st.title("📖 Model Documentation")

    st.markdown("""
    ## How It Works

    This predictor uses a **Poisson regression** model — the standard approach in football
    analytics for modelling goal-scoring — trained on historical international match data
    and applied to the 48-team WC 2026 field.

    ### Core Formula

    For each team in a match, goals scored are modelled as:

    > **Goals ~ Poisson(λ)**

    Where λ (expected goals) is:

    > **λ = exp(β₀ + β_elo·elo_diff + β_rank·ranking_diff + β_home·is_home + β_form·form_scored + β_squad·squad_rating_diff + β_rd·rd_combined)**

    Both teams' goals are modelled independently, giving a full joint probability matrix
    over any scoreline. Win/Draw/Loss probabilities are derived from this matrix.
    A **Dixon-Coles correction** (ρ = −0.13) adjusts probabilities for low-scoring outcomes.

    ### Features & Learned Coefficients

    | Feature | Description | Coefficient |
    |---|---|---|
    | `elo_diff` | Glicko-1 rating difference | β = 0.0001 |
    | `ranking_diff` | FIFA ranking difference | β = −0.0065 |
    | `is_home` | Home advantage (0 for all WC matches) | β = 0.2517 |
    | `form_scored` | Rolling avg goals scored, last 10 matches | β = 0.0705 |
    | `squad_rating_diff` | EA FC squad average overall rating difference | β = 0.0209 |
    | `rd_combined` | Average Glicko rating deviation — shrinks predictions toward 50/50 for uncertain teams | β = −0.0001 |

    ### Training

    - **Data**: 50,000+ international matches since 2000 (martj42 Kaggle dataset)
    - **Ratings**: Glicko-1 system, which adds a rating deviation (RD) to classic Elo — teams
      with high RD (less match history) are treated as more uncertain and their predictions
      pulled slightly toward 50/50
    - **Weighting**: Matches weighted by tournament importance × recency decay
    - **Confederation offsets**: Applied at prediction time to correct for Elo inflation
      from within-confederation play (CONCACAF −100, AFC −80, CAF −50)

    ### Tournament Simulation

    The Tournament Odds page runs 10,000 Monte Carlo simulations of the full WC 2026 format:
    - Group stage: 12 groups × 4 teams, 6 matches each
    - Top 2 per group + 8 best 3rd-place teams = 32 teams in Round of 32
    - Knockout rounds through to the Final; draws go to a 50/50 penalty shootout

    The Group Stage Predictor shows the model's single most likely scoreline for each of the
    72 group matches, with predicted final standings.

    ### Known Limitations

    - No injury or suspension data
    - No actual lineup or formation data
    - EA FC ratings underestimate some non-European players
    - Penalty shootouts modelled as 50/50 (no historical shootout data)
    """)

    elo_df = load_elo()
    if elo_df is not None:
        st.divider()
        from data_loader import WC2026_GROUPS
        wc_teams = {t for teams in WC2026_GROUPS.values() for t in teams}
        st.subheader("Current Glicko Ratings — WC 2026 Teams")
        wc_elo = elo_df[elo_df["team"].isin(wc_teams)].copy()
        if "rd" in wc_elo.columns:
            wc_elo["rd"] = wc_elo["rd"].round(1)
        wc_elo["elo"] = wc_elo["elo"].round(1)
        st.dataframe(wc_elo, use_container_width=True, hide_index=True)
