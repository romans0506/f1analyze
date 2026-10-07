import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1app.charts import style, team_color
from views.common import get_data, get_prediction

data = get_data()
upcoming = data.upcoming_races()
if upcoming.empty:
    st.info("No upcoming races on the calendar — the season is over.")
    st.stop()

labels = {r.raceId: f"R{r.round} · {r.name} ({r.date:%d %b %Y})" for r in upcoming.itertuples()}
race_id = st.sidebar.selectbox("Upcoming race", list(labels), format_func=labels.get)
race = upcoming.set_index("raceId").loc[race_id]
sim, meta = get_prediction(int(race_id))

st.title(f"🔮 {race['name']} {race.year}")
st.caption(f"{race.circuit} · {race.location}, {race.country} · race day {race.date:%A %d %B %Y}"
           + (" · sprint weekend" if race.has_sprint else ""))

if meta["uses_grid"]:
    st.success("Qualifying is done — this prediction uses the **real starting grid**.")
else:
    st.info("Qualifying hasn't happened yet, so this is a **pre-qualifying** prediction based on form, "
            "team pace and track history. It sharpens automatically once the grid is known.")
if race_id != upcoming.raceId.iloc[0]:
    st.warning("This race is more than one round away; the prediction uses today's form.")

fav, pod = sim.iloc[sim.p_win.idxmax()], sim.iloc[sim.p_podium.idxmax()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Favourite", fav.driver, f"{fav.p_win:.0%} win chance", delta_color="off")
c2.metric("Most likely on the podium", pod.driver, f"{pod.p_podium:.0%}", delta_color="off")
c3.metric("Simulations", "20,000")
c4.metric("Expected retirements", f"{sim.p_dnf.sum():.1f} cars")

colors = {r.code: team_color(r.constructorRef) for r in sim.itertuples()}

# ------------------------------------------------------------- probabilities
st.subheader("Win, podium and points chances")
metric = st.radio("Show", ["Win", "Podium", "Points (top 10)"], horizontal=True, label_visibility="collapsed")
col = {"Win": "p_win", "Podium": "p_podium", "Points (top 10)": "p_points"}[metric]
ranked = sim.sort_values(col, ascending=True)
fig = px.bar(ranked, x=col, y="code", orientation="h", color="code", color_discrete_map=colors,
             hover_data={"driver": True, "team": True, col: ":.1%", "code": False},
             labels={col: f"{metric} probability", "code": ""})
fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
st.plotly_chart(style(fig, 620), width="stretch")

# -------------------------------------------------------- predicted order
left, right = st.columns([1, 1])
with left:
    st.subheader("Predicted finishing order")
    st.caption("Dot = expected position, bar = 80% range across simulations.")
    order = sim.sort_values("predicted_pos", ascending=False)
    fig = go.Figure()
    for r in order.itertuples():
        fig.add_trace(go.Scatter(
            x=[r.pos_p10, r.pos_p90], y=[r.code, r.code], mode="lines",
            line=dict(color=colors[r.code], width=6), opacity=0.35, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[r.expected_pos], y=[r.code], mode="markers", marker=dict(color=colors[r.code], size=12),
            name=r.driver, showlegend=False,
            hovertemplate=f"{r.driver}<br>expected P{r.expected_pos:.1f}<br>80% range P{r.pos_p10:.0f}–P{r.pos_p90:.0f}<extra></extra>"))
    fig.update_xaxes(title="Finishing position", autorange="reversed", dtick=2)
    st.plotly_chart(style(fig, 620), width="stretch")

with right:
    st.subheader("Where each driver finishes")
    st.caption("Probability of every finishing position (rows sorted by predicted order).")
    dist = np.vstack(sim.position_dist.to_numpy())
    fig = px.imshow(dist, x=[f"P{i}" for i in range(1, dist.shape[1] + 1)], y=sim.code.tolist(),
                    color_continuous_scale="Reds", aspect="auto", labels=dict(color="Probability"))
    fig.update_traces(hovertemplate="%{y} finishes %{x}: %{z:.1%}<extra></extra>")
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(style(fig, 620), width="stretch")

# ------------------------------------------------------------- the "why"
st.subheader("What the model is looking at")
feature_names = {
    "grid": "Grid", "drv_form": "Avg finish (last 5)", "drv_pts": "Avg points (last 5)",
    "drv_quali": "Avg quali (last 5)", "drv_dnf": "DNF rate (last 10)", "team_form": "Team avg finish",
    "team_quali": "Team avg quali", "track_hist": "Avg finish here", "champ_rank": "Championship pos",
    "sprint_pos": "Sprint result",
}
table = sim[["predicted_pos", "driver", "team", "p_win", "p_podium", "p_points", "p_dnf"]
            + [f for f in meta["features"] if f in sim.columns]]
table = table.rename(columns={"predicted_pos": "Pred.", "driver": "Driver", "team": "Team", "p_win": "Win",
                              "p_podium": "Podium", "p_points": "Points", "p_dnf": "DNF risk", **feature_names})
pct = {c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
       for c in ["Win", "Podium", "Points", "DNF risk"]}
st.dataframe(table, hide_index=True, width="stretch",
             column_config={**pct, **{v: st.column_config.NumberColumn(format="%.1f") for v in feature_names.values()}})

# ------------------------------------------------------------- track history
st.subheader(f"History at {race.circuit}")
past = data.results[(data.results.circuitRef == race.circuitRef)]
if past.empty:
    st.write("No previous Grands Prix at this circuit in the database.")
else:
    a, b = st.columns(2)
    with a:
        winners = (past[past.positionOrder == 1].sort_values("date", ascending=False).head(10)
                   [["year", "driver", "team", "grid"]].rename(columns=str.title))
        st.markdown("**Last 10 winners**")
        st.dataframe(winners, hide_index=True, width="stretch")
    with b:
        field = past[past.driverRef.isin(sim.driverRef)]
        hist = (field.groupby("code").agg(starts=("raceId", "nunique"), wins=("positionOrder", lambda s: (s == 1).sum()),
                                          podiums=("positionOrder", lambda s: (s <= 3).sum()),
                                          best=("positionOrder", "min"))
                .sort_values(["wins", "podiums", "best"], ascending=[False, False, True]).reset_index())
        st.markdown("**Current drivers here**")
        st.dataframe(hist.rename(columns=str.title), hide_index=True, width="stretch")

st.caption(f"Model: gradient-boosted trees on finishing position since 2014 (recent seasons weighted more), "
           f"plus Monte Carlo with retirements (baseline DNF rate {meta['dnf_prior']:.0%}). "
           "See **Model accuracy** for how it would have done on past races.")
