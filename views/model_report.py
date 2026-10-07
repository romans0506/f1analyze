import plotly.express as px
import streamlit as st

from f1app.charts import style
from f1app.model import summarize_backtest
from views.common import get_backtest, get_data

data = get_data()
st.title("📐 How good are the predictions?")
st.markdown(
    "Walk-forward backtest: for every race below, the model is trained **only on races before it**, "
    "then predicts it — exactly what happens for the next GP. The baseline is the simplest possible "
    "forecast: *the starting grid is the finishing order*.")

latest = int(data.completed_races().year.max())
seasons = st.multiselect("Seasons to test", list(range(latest, 2015, -1)), default=[latest, latest - 1])
mode = st.radio("Prediction made", ["After qualifying (grid known)", "Before qualifying"], horizontal=True)
if not seasons:
    st.stop()
bt = get_backtest(tuple(sorted(seasons)), mode.startswith("After"))
s = summarize_backtest(bt)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Races tested", s["races"])
c2.metric("Winner picked correctly", f"{s['winner_accuracy']:.0%}",
          f"{s['winner_accuracy'] - s['pole_baseline_accuracy']:+.0%} vs pole-sitter", delta_color="normal")
c3.metric("Podium finishers named", f"{s['avg_podium_hits']:.2f} / 3")
c4.metric("Order correlation (Spearman)", f"{s['avg_spearman']:.2f}",
          f"{s['avg_spearman'] - s['grid_baseline_spearman']:+.2f} vs grid order")
st.caption(f"On average the model gave the eventual winner a {s['avg_p_win_actual_winner']:.0%} win chance "
           f"(a random pick among ~20 drivers would be 5%).")

st.subheader("Race by race")
bt = bt.sort_values(["year", "round"])
bt["race"] = bt.year.astype(str) + " R" + bt["round"].astype(str) + " " + bt.name.str.replace(" Grand Prix", "")
long = bt.melt(id_vars="race", value_vars=["spearman", "grid_spearman"], var_name="forecast", value_name="corr")
long["forecast"] = long.forecast.map({"spearman": "Model", "grid_spearman": "Grid order"})
fig = px.line(long, x="race", y="corr", color="forecast", markers=True,
              color_discrete_map={"Model": "#E10600", "Grid order": "#888888"},
              labels={"corr": "Rank correlation with actual result", "race": "", "forecast": ""})
fig.update_xaxes(tickangle=-60)
st.plotly_chart(style(fig, 460), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Win chance given to the actual winner")
    fig = px.histogram(bt, x="p_win_real_winner", nbins=10, labels={"p_win_real_winner": "Model's win probability"})
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 360), width="stretch")
with right:
    st.subheader("Podium finishers named correctly")
    counts = bt.podium_hits.value_counts().reindex(range(4), fill_value=0).rename_axis("hits").reset_index(name="races")
    fig = px.bar(counts, x="hits", y="races", labels={"hits": "Correct podium finishers (of 3)", "races": "Races"})
    st.plotly_chart(style(fig, 360), width="stretch")

st.dataframe(bt[["race", "winner_pred", "winner_real", "p_win_real_winner", "podium_hits", "spearman"]]
             .rename(columns={"race": "Race", "winner_pred": "Predicted winner", "winner_real": "Actual winner",
                              "p_win_real_winner": "P(win) of actual winner", "podium_hits": "Podium hits",
                              "spearman": "Rank corr."}),
             hide_index=True, width="stretch",
             column_config={"P(win) of actual winner": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                            "Rank corr.": st.column_config.NumberColumn(format="%.2f")})

with st.expander("How the model works"):
    st.markdown("""
- **Features** (all computed from races *before* the one being predicted): the driver's average finish, points and
  qualifying position over the last 5 races; retirement rate over the last 10; the team's average finish and qualifying;
  the driver's past results at this circuit; championship position; the sprint result on sprint weekends; and, once
  qualifying is done, the starting grid.
- **Model**: gradient-boosted decision trees predicting finishing position for drivers who finish, trained on every
  race since 2014 with each earlier season weighted 30% less than the one after it.
- **Simulation**: each race is run 20,000 times. Each driver's result is their prediction plus random race-day noise
  (larger further down the order), and cars retire at random based on their recent reliability.
  Counting outcomes gives the win, podium and points probabilities.
- **Limits**: the model doesn't know about weather, grid penalties announced after qualifying, upgrades or crashes in
  practice, so treat it as an informed baseline, not a certainty.
""")
