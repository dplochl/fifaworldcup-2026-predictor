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

@st.cache_data
def load_matches():
    p = PROCESSED / "matches_processed.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

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
    wc_page = st.radio("", [
        "🏆 Tournament Odds",
        "🔍 Model vs. FIFA Rankings",
        "📊 Group Tables",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("## 🌍 International Predictor")
    intl_page = st.radio("", [
        "⚔️ Match Predictor",
        "👤 Team Profile",
        "📖 Model Info",
    ], label_visibility="collapsed")

# Determine active page (last interacted radio wins — use session state trick)
if "last_section" not in st.session_state:
    st.session_state.last_section = "wc"

# Simple priority: if user clicks a WC option we show that; if intl, we show that.
# We track which was changed by checking against stored values.
if "prev_wc"   not in st.session_state: st.session_state.prev_wc   = wc_page
if "prev_intl" not in st.session_state: st.session_state.prev_intl = intl_page

if wc_page != st.session_state.prev_wc:
    st.session_state.last_section = "wc"
    st.session_state.prev_wc = wc_page
if intl_page != st.session_state.prev_intl:
    st.session_state.last_section = "intl"
    st.session_state.prev_intl = intl_page

page = wc_page if st.session_state.last_section == "wc" else intl_page


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
    from data_loader import (WC2026_GROUPS, WC2026_R32_BRACKET,
                              WC2026_THIRD_PLACE_ELIGIBLE, WC2026_R16_PAIRS,
                              WC2026_QF_PAIRS, WC2026_SF_PAIRS,
                              TEAM_FLAGS, HOST_NATIONS)
    from itertools import combinations

    st.title("⚽ Full Tournament Predictor")
    st.caption("Model's most likely outcome for every match — group stage through the final")

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

    def _decisive_score(p):
        """For knockout matches: most likely score conditioned on the most likely outcome.
        Avoids the Poisson mode landing on a draw for every near-even matchup — the global
        score mode is often 1-1 even when one team is clearly favoured in aggregate."""
        matrix = p.get("score_matrix")
        if matrix is None:
            return map(int, p["most_likely_score"].split("–"))
        n = min(matrix.shape[0], 9)
        m = matrix[:n, :n]
        rows, cols = np.indices((n, n))
        ph, pd_val, pa = p["prob_home_win"], p["prob_draw"], p["prob_away_win"]
        if ph >= pa and ph >= pd_val:
            mask = rows > cols      # home-win scorelines
        elif pa > ph and pa >= pd_val:
            mask = cols > rows      # away-win scorelines
        else:
            mask = rows == cols     # draw scorelines → will trigger penalties
        r, c = np.unravel_index(np.where(mask, m, 0).argmax(), m.shape)
        return int(r), int(c)

    def predict_game(t1, t2, knockout=False):
        """Predict a match; handles host nation home advantage for group stage."""
        if not knockout and t2 in HOST_NATIONS and t1 not in HOST_NATIONS:
            p = predict(t2, t1, elo_dict, form, squad, rd_dict, neutral=False)
            if not p:
                return None
            hg, ag = _decisive_score(p) if knockout else map(int, p["most_likely_score"].split("–"))
            return dict(t1=t1, t2=t2, g1=ag, g2=hg,
                        p1=p["prob_away_win"], pd=p["prob_draw"], p2=p["prob_home_win"])
        if not knockout and t1 in HOST_NATIONS:
            p = predict(t1, t2, elo_dict, form, squad, rd_dict, neutral=False)
        else:
            p = predict(t1, t2, elo_dict, form, squad, rd_dict, neutral=True)
        if not p:
            return None
        hg, ag = _decisive_score(p) if knockout else map(int, p["most_likely_score"].split("–"))
        return dict(t1=t1, t2=t2, g1=hg, g2=ag,
                    p1=p["prob_home_win"], pd=p["prob_draw"], p2=p["prob_away_win"])

    def ko_winner(m):
        """Return (winner, went_to_pens). Draw → higher win-prob team advances."""
        if m["g1"] > m["g2"]:
            return m["t1"], False
        if m["g2"] > m["g1"]:
            return m["t2"], False
        return (m["t1"] if m["p1"] >= m["p2"] else m["t2"]), True

    def match_card(m, show_probs=True, ko=False):
        """Render a match result as an HTML card."""
        if ko:
            w, pens = ko_winner(m)
        else:
            g1, g2 = m["g1"], m["g2"]
            w = m["t1"] if g1 > g2 else (m["t2"] if g2 > g1 else None)
            pens = False
        t1_style = "font-weight:700;color:#C4981F" if w == m["t1"] else "color:#E8E8E8"
        t2_style = "font-weight:700;color:#C4981F" if w == m["t2"] else "color:#E8E8E8"
        score = f"{m['g1']}–{m['g2']}" + (" (pens)" if pens else "")
        prob_row = ""
        if show_probs:
            prob_row = (f"<div style='font-size:0.68rem;color:#8A9BC0;"
                        f"text-align:center;margin-top:3px'>"
                        f"{m['p1']*100:.0f}% · {m['pd']*100:.0f}% · {m['p2']*100:.0f}%</div>")
        return (f"<div style='background:linear-gradient(135deg,#0E1628,#121E38);"
                f"border:1px solid #1E2F50;border-radius:8px;padding:10px 14px;margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:6px'>"
                f"<span style='{t1_style};flex:1'>{flag_t(m['t1'])}</span>"
                f"<span style='color:#C4A030;font-weight:700;font-size:1rem;"
                f"white-space:nowrap'>{score}</span>"
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

    # ── compute all predictions (cached in session state per elo snapshot) ────
    cache_key = f"tourn_{round(elo_df['elo'].sum())}"
    if cache_key not in st.session_state:
        with st.spinner("Running model predictions for all 103 matches…"):

            # Group stage
            gp = {}
            for group, teams in WC2026_GROUPS.items():
                matches = []
                for t1, t2 in combinations(teams, 2):
                    m = predict_game(t1, t2)
                    if m:
                        matches.append(m)
                gp[group] = matches

            # Standings + qualification
            standings, thirds, qualifiers = {}, [], {}
            for group, teams in WC2026_GROUPS.items():
                ranked, tbl = compute_standings(gp[group], teams)
                standings[group] = {"ranked": ranked, "table": tbl}
                qualifiers[group] = {"W": ranked[0], "R": ranked[1]}
                t3 = ranked[2]
                thirds.append(dict(team=t3, group=group,
                                   pts=tbl[t3]["Pts"], gd=tbl[t3]["GD"], gf=tbl[t3]["GF"]))

            thirds.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
            avail = list(thirds[:8])
            third_assign = {}
            for slot in [74, 77, 79, 80, 81, 82, 85, 87]:
                elig = WC2026_THIRD_PLACE_ELIGIBLE.get(slot, set())
                assigned = False
                for i, t in enumerate(avail):
                    if t["group"] in elig:
                        third_assign[slot] = t["team"]
                        avail.pop(i)
                        assigned = True
                        break
                if not assigned and avail:
                    third_assign[slot] = avail.pop(0)["team"]

            def resolve(slot):
                k, ref = slot
                if k == "W":
                    return qualifiers[ref]["W"]
                if k == "R":
                    return qualifiers[ref]["R"]
                return third_assign.get(ref, "TBD")

            # R32
            r32 = []
            for mnum, s1, s2 in WC2026_R32_BRACKET:
                t1, t2 = resolve(s1), resolve(s2)
                m = predict_game(t1, t2, knockout=True)
                if m:
                    w, pens = ko_winner(m)
                    r32.append({**m, "winner": w, "pens": pens, "match_num": mnum})

            def sim_ko_round(pairs, prev):
                result = []
                for i, j in pairs:
                    t1, t2 = prev[i]["winner"], prev[j]["winner"]
                    m = predict_game(t1, t2, knockout=True)
                    if m:
                        w, pens = ko_winner(m)
                        result.append({**m, "winner": w, "pens": pens})
                return result

            r16   = sim_ko_round(WC2026_R16_PAIRS, r32)
            qf    = sim_ko_round(WC2026_QF_PAIRS,  r16)
            sf    = sim_ko_round(WC2026_SF_PAIRS,   qf)

            final_t1, final_t2 = sf[0]["winner"], sf[1]["winner"]
            final_m = predict_game(final_t1, final_t2, knockout=True)
            if final_m:
                champ, final_pens = ko_winner(final_m)
                final_m.update(winner=champ, pens=final_pens)

            st.session_state[cache_key] = dict(
                gp=gp, standings=standings, thirds=thirds,
                r32=r32, r16=r16, qf=qf, sf=sf,
                final_m=final_m, champion=final_m["winner"] if final_m else "?"
            )

    data      = st.session_state[cache_key]
    gp        = data["gp"]
    standings = data["standings"]
    thirds    = data["thirds"]
    r32       = data["r32"]
    r16       = data["r16"]
    qf        = data["qf"]
    sf        = data["sf"]
    final_m   = data["final_m"]
    champion  = data["champion"]

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["⚽ Group Matches", "📊 Group Standings", "🔄 Round of 32", "🏆 Knockouts"]
    )

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
                        st.markdown(match_card(m, show_probs=True), unsafe_allow_html=True)
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

    # ── Tab 3: Round of 32 ────────────────────────────────────────────────────
    with tab3:
        st.caption("All 16 matches of the Round of 32 · (pens) = decided on penalties")
        c1, c2 = st.columns(2)
        for i, m in enumerate(r32):
            col = c1 if i < 8 else c2
            col.markdown(match_card(m, show_probs=True, ko=True), unsafe_allow_html=True)

    # ── Tab 4: Knockouts ──────────────────────────────────────────────────────
    with tab4:
        # Champion banner
        st.markdown(
            f"<div style='text-align:center;background:linear-gradient(135deg,#1A0F00,#2A1A00);"
            f"border:2px solid #C4981F;border-radius:16px;padding:20px;margin-bottom:24px'>"
            f"<div style='color:#C4981F;font-family:\"Bebas Neue\",sans-serif;"
            f"font-size:1.1rem;letter-spacing:3px'>PREDICTED CHAMPION</div>"
            f"<div style='font-size:2.5rem;margin:6px 0'>{TEAM_FLAGS.get(champion, '')}</div>"
            f"<div style='color:#C4A030;font-family:\"Bebas Neue\",sans-serif;"
            f"font-size:2rem;letter-spacing:2px'>{champion}</div></div>",
            unsafe_allow_html=True)

        st.subheader("Round of 16")
        c1, c2 = st.columns(2)
        for i, m in enumerate(r16):
            (c1 if i < 4 else c2).markdown(
                match_card(m, show_probs=False, ko=True), unsafe_allow_html=True)

        st.divider()
        st.subheader("Quarter-finals")
        c1, c2 = st.columns(2)
        for i, m in enumerate(qf):
            (c1 if i < 2 else c2).markdown(
                match_card(m, show_probs=False, ko=True), unsafe_allow_html=True)

        st.divider()
        st.subheader("Semi-finals")
        c1, c2 = st.columns(2)
        for i, m in enumerate(sf):
            (c1 if i == 0 else c2).markdown(
                match_card(m, show_probs=False, ko=True), unsafe_allow_html=True)

        st.divider()
        st.subheader("Final")
        _, c_mid, _ = st.columns([1, 2, 1])
        c_mid.markdown(match_card(final_m, show_probs=True, ko=True), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# International Predictor — Page 1: Match Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Match Predictor":
    st.title("Head-to-Head Match Predictor")
    st.caption("Enter two teams to get predicted probabilities and expected scoreline")

    elo_df = load_elo()
    model  = load_model()

    if elo_df is None or model is None:
        st.warning("⚠️ Run the data pipeline first (`data_loader.py` → `model.py`)")
    else:
        from data_loader import FIFA_RANKINGS, TEAM_FLAGS
        teams         = sorted([t for t in elo_df["team"].tolist() if t in FIFA_RANKINGS])
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
# International Predictor — Page 2: Team Profile
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Team Profile":
    st.title("👤 Team Profile")
    st.caption("Elo history, recent results, and model statistics for any team")

    from data_loader import FIFA_RANKINGS, TEAM_FLAGS, TEAM_CONFEDERATION

    elo_df   = load_elo()
    matches  = load_matches()

    if elo_df is None or matches is None:
        st.warning("⚠️ Run `python src/data_loader.py` first.")
        st.stop()

    all_teams   = sorted(elo_df["team"].tolist())
    flag_teams  = [f"{TEAM_FLAGS.get(t,'')} {t}" for t in all_teams]
    team_map    = {f"{TEAM_FLAGS.get(t,'')} {t}": t for t in all_teams}

    default_t   = "🇦🇹 Austria" if "🇦🇹 Austria" in flag_teams else flag_teams[0]
    sel_display = st.selectbox("Select a team", flag_teams,
                               index=flag_teams.index(default_t) if default_t in flag_teams else 0)
    team        = team_map[sel_display]

    st.divider()

    # ── Key metrics ───────────────────────────────────────────────────────────
    elo_row  = elo_df[elo_df["team"] == team]
    elo_val  = round(elo_row["elo"].values[0], 1) if not elo_row.empty else "N/A"
    rd_val   = round(elo_row["rd"].values[0], 1)  if not elo_row.empty and "rd" in elo_row.columns else "N/A"
    fifa_r   = FIFA_RANKINGS.get(team, "Not ranked")
    conf     = TEAM_CONFEDERATION.get(team, "Unknown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Glicko Rating", elo_val)
    c2.metric("Rating Deviation", rd_val, help="Lower = more certain about rating")
    c3.metric("FIFA Ranking", f"#{fifa_r}" if isinstance(fifa_r, int) else fifa_r)
    c4.metric("Confederation", conf)

    st.divider()

    # ── Elo history chart ─────────────────────────────────────────────────────
    st.subheader("Rating History")
    home_hist = matches[matches["home_team"] == team][["date", "elo_home"]].rename(columns={"elo_home": "elo"})
    away_hist = matches[matches["away_team"] == team][["date", "elo_away"]].rename(columns={"elo_away": "elo"})
    hist      = pd.concat([home_hist, away_hist]).sort_values("date").drop_duplicates("date")

    if not hist.empty:
        fig = go.Figure(go.Scatter(
            x=hist["date"], y=hist["elo"],
            mode="lines", line=dict(color="#C4981F", width=2),
            fill="tozeroy", fillcolor="rgba(255,182,39,0.07)",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2a40"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rating history available.")

    st.divider()

    # ── Last 10 matches ───────────────────────────────────────────────────────
    st.subheader("Last 10 Matches")
    home_m = matches[matches["home_team"] == team].copy()
    home_m["opponent"] = home_m["away_team"]
    home_m["scored"]   = home_m["home_score"]
    home_m["conceded"] = home_m["away_score"]
    home_m["venue"]    = "Home"

    away_m = matches[matches["away_team"] == team].copy()
    away_m["opponent"] = away_m["home_team"]
    away_m["scored"]   = away_m["away_score"]
    away_m["conceded"] = away_m["home_score"]
    away_m["venue"]    = "Away"

    all_m = pd.concat([home_m, away_m]).sort_values("date", ascending=False).head(10)

    if all_m.empty:
        st.info("No match history found.")
    else:
        def result_str(row):
            if row["scored"] > row["conceded"]:  return "✅ W"
            if row["scored"] == row["conceded"]: return "🟡 D"
            return "❌ L"

        display = pd.DataFrame({
            "Date":       all_m["date"].dt.strftime("%Y-%m-%d"),
            "Opponent":   all_m["opponent"].map(lambda t: f"{TEAM_FLAGS.get(t,'')} {t}"),
            "Score":      all_m.apply(lambda r: f"{int(r['scored'])}–{int(r['conceded'])}", axis=1),
            "Result":     all_m.apply(result_str, axis=1),
            "Venue":      all_m["venue"],
            "Tournament": all_m.get("tournament", pd.Series([""] * len(all_m))),
        })
        st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# International Predictor — Page 3: Model Info
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Model Info":
    st.title("📖 Model Documentation")

    st.markdown("""
    ## How It Works

    This predictor uses a **Poisson regression** model — the standard approach in football
    analytics for modelling goal-scoring — enhanced with several features derived from
    historical international match data.

    ### Core Formula

    For each team in a match, we model goals scored as:

    > **Goals ~ Poisson(λ)**

    Where λ (expected goals) is:

    > **λ = exp(β₀ + β_elo·elo_diff + β_rank·ranking_diff + β_home·is_home + β_form·form_scored + β_squad·squad_rating_diff)**

    Both teams' goals are modelled independently, giving a full joint probability matrix
    over any scoreline. Win/Draw/Loss probabilities are derived from this matrix.
    A **Dixon-Coles correction** (ρ = −0.13) adjusts probabilities for 0-0 and 1-1 draws.

    ### Features

    | Feature | Description |
    |---|---|
    | `elo_diff` | Glicko rating difference (accounts for uncertainty via RD) |
    | `ranking_diff` | FIFA ranking difference |
    | `is_home` | 1 if the team is playing at home (0 for all WC matches) |
    | `form_scored` | Rolling average goals scored over last 10 matches |
    | `squad_rating_diff` | EA FC squad average overall rating difference |

    ### Training

    - **Data**: 50,000+ international matches since 2000 (martj42 Kaggle dataset)
    - **Ratings**: Glicko-1 (replaces Elo — adds rating deviation for uncertainty)
    - **Weighting**: Matches weighted by tournament importance × recency decay
    - **Confederation offsets**: Applied at prediction time to correct for Elo inflation
      from within-confederation play (CONCACAF −100, AFC −80, CAF −50)

    ### Tournament Simulation

    The 10,000 Monte Carlo simulations model the full 48-team WC 2026 format:
    - Group stage (12 groups × 4 teams)
    - Top 2 per group + 8 best 3rd-place teams = 32 in R32
    - Knockout rounds to Final; draws resolved by 50/50 penalty shootout

    ### Known Limitations

    - No injury or suspension data
    - No actual lineup / formation data
    - EA FC ratings underestimate some non-European players
    - Penalty shootout modelled as 50/50 (no historical shootout records)
    """)

    elo_df = load_elo()
    if elo_df is not None:
        st.divider()
        st.subheader("Current Glicko Ratings — Top 30")
        top30 = elo_df.head(30).copy()
        if "rd" in top30.columns:
            top30["rd"] = top30["rd"].round(1)
        top30["elo"] = top30["elo"].round(1)
        st.dataframe(top30, use_container_width=True, hide_index=True)
