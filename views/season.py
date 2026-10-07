import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from f1app.charts import driver_colors, driver_dashes, style, team_color
from views.common import get_data, season_picker

data = get_data()
year = season_picker(data, key="season_year")
st.title(f"🏆 {year} season")

res = data.results[data.results.year == year]
spr = data.sprints[data.sprints.year == year]
rounds = data.completed_races().query("year == @year").sort_values("round")
total_rounds = int(data.races.query("year == @year")["round"].max())
st.caption(f"{len(rounds)} of {total_rounds} rounds completed · last: {rounds.iloc[-1]['name']}")

# Points per driver per round, including sprints.
pts = pd.concat([res[["round", "code", "driver", "constructorRef", "team", "points"]],
                 spr[["round", "code", "driver", "constructorRef", "team", "points"]]])

# ------------------------------------------------------------- standings
drivers = (res.groupby(["code", "driver"])
              .agg(wins=("positionOrder", lambda s: (s == 1).sum()),
                   podiums=("positionOrder", lambda s: (s <= 3).sum()),
                   dnfs=("finished", lambda s: (~s.astype(bool)).sum()),
                   avg_finish=("positionOrder", "mean"))
              .join(pts.groupby("code").points.sum())
              .join(pts.drop_duplicates("code", keep="last").set_index("code")[["team", "constructorRef"]])
              .reset_index().sort_values(["points", "wins"], ascending=False))
drivers.insert(0, "pos", np.arange(1, len(drivers) + 1))
teams = (pts.groupby(["constructorRef", "team"]).points.sum().reset_index()
            .drop_duplicates("constructorRef", keep="last").sort_values("points", ascending=False))
leader, second = drivers.iloc[0], drivers.iloc[1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Championship leader", leader.driver, f"{leader.points:.0f} pts", delta_color="off")
c2.metric("Gap to 2nd", f"{leader.points - second.points:.0f} pts", second.driver, delta_color="off")
c3.metric("Leading team", teams.iloc[0].team, f"{teams.iloc[0].points:.0f} pts", delta_color="off")
c4.metric("Different winners", int((drivers.wins > 0).sum()))

left, right = st.columns([3, 2])
with left:
    st.subheader("Drivers' championship")
    st.dataframe(drivers[["pos", "driver", "team", "points", "wins", "podiums", "dnfs", "avg_finish"]]
                 .rename(columns={"pos": "Pos", "driver": "Driver", "team": "Team", "points": "Points", "wins": "Wins",
                                  "podiums": "Podiums", "dnfs": "DNFs", "avg_finish": "Avg finish"}),
                 hide_index=True, width="stretch", height=560,
                 column_config={"Avg finish": st.column_config.NumberColumn(format="%.1f"),
                                "Points": st.column_config.NumberColumn(format="%.0f")})
with right:
    st.subheader("Constructors' championship")
    fig = px.bar(teams.iloc[::-1], x="points", y="team", orientation="h", color="constructorRef",
                 color_discrete_map={t: team_color(t) for t in teams.constructorRef}, text="points",
                 labels={"points": "Points", "team": ""})
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, 560), width="stretch")

# ------------------------------------------------------------- progression
st.subheader("Points progression")
top_n = st.slider("Drivers shown", 2, len(drivers), min(10, len(drivers)))
show = drivers.code.head(top_n).tolist()
prog = (pts[pts.code.isin(show)].groupby(["round", "code"]).points.sum().unstack(fill_value=0)
           .reindex(rounds["round"], fill_value=0).cumsum().stack().rename("points").reset_index())
prog["race"] = prog["round"].map(rounds.set_index("round")["name"].str.replace(" Grand Prix", ""))
fig = px.line(prog, x="round", y="points", color="code", line_dash="code", markers=True,
              color_discrete_map=driver_colors(pts, "code"), category_orders={"code": show},
              line_dash_map=driver_dashes(pts, "code"), hover_data=["race"],
              labels={"round": "Round", "points": "Cumulative points", "code": ""})
st.plotly_chart(style(fig, 500), width="stretch")

# ------------------------------------------------------------- results grid
st.subheader("Every result")
st.caption("Finishing position per round (✕ = did not finish). Hover for details.")
grid = res.pivot_table(index="code", columns="round", values="positionOrder", aggfunc="first").reindex(drivers.code)
dnf = res.pivot_table(index="code", columns="round", values="finished", aggfunc="first").reindex(drivers.code)
text = grid.map(lambda v: "" if pd.isna(v) else f"{v:.0f}").where(dnf.astype("boolean").fillna(True).astype(bool), "✕")
short = rounds.set_index("round")["name"].str.replace(" Grand Prix", "").str.slice(0, 12)
fig = px.imshow(grid, color_continuous_scale="RdYlGn_r", aspect="auto",
                x=[f"R{r} {short.get(r, '')}" for r in grid.columns], y=grid.index,
                labels=dict(color="Position"))
fig.update_traces(text=text.values, texttemplate="%{text}", hovertemplate="%{y} · %{x}: P%{z}<extra></extra>")
fig.update_xaxes(side="top", tickangle=-45)
st.plotly_chart(style(fig, 40 + 24 * len(grid)), width="stretch")

# ------------------------------------------------------------- team points per race
st.subheader("Team points per round")
tp = pts.groupby(["round", "constructorRef", "team"]).points.sum().reset_index()
fig = px.bar(tp, x="round", y="points", color="constructorRef",
             color_discrete_map={t: team_color(t) for t in tp.constructorRef.unique()},
             hover_data=["team"], labels={"round": "Round", "points": "Points", "constructorRef": "Team"})
st.plotly_chart(style(fig, 420), width="stretch")
