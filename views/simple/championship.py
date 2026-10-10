import pandas as pd
import plotly.express as px
import streamlit as st

from f1app.charts import style, team_color
from views.common import get_data
from views.simple.widgets import podium

MAX_RACE_PTS, MAX_SPRINT_PTS = 25, 8

data = get_data()
year = int(data.completed_races().year.max())
st.title(f"🏆 {year} Championship")

res = data.results[data.results.year == year]
spr = data.sprints[data.sprints.year == year]
pts = pd.concat([res, spr])[["round", "driverRef", "driver", "code", "constructorRef", "team", "points"]]
standings = (pts.groupby("driverRef").points.sum().sort_values(ascending=False).rename("points").reset_index()
             .merge(pts.drop_duplicates("driverRef", keep="last")[["driverRef", "driver", "code", "constructorRef", "team"]],
                    on="driverRef"))

schedule = data.races[data.races.year == year]
done_ids = set(res.raceId)
left = schedule[~schedule.raceId.isin(done_ids)]
max_left = len(left) * MAX_RACE_PTS + int(left.has_sprint.sum()) * MAX_SPRINT_PTS

podium([{"code": r.code, "name": r.driver, "team": r.team, "ref": r.constructorRef,
         "line": f"{r.points:.0f} points"} for r in standings.head(3).itertuples()],
       caption=f"After {len(done_ids)} of {len(schedule)} races")

# ------------------------------------------------------------- who can still win
st.header("Who can still be champion?")
lead = standings.points.iloc[0]
alive = standings[standings.points + max_left >= lead]
if left.empty:
    st.success(f"🏆 **{standings.driver.iloc[0]}** is the {year} World Champion!")
else:
    st.markdown(f"There are **{len(left)} races** left"
                + (f" (plus {int(left.has_sprint.sum())} sprint{'s' if left.has_sprint.sum() > 1 else ''})"
                   if left.has_sprint.any() else "")
                + f", so at most **{max_left} points** are still up for grabs. "
                  "A driver can still win the title if they could catch the leader by winning everything from here.")
    cols = st.columns(min(4, len(alive)))
    for i, r in enumerate(alive.head(4).itertuples()):
        behind = lead - r.points
        cols[i].metric(r.driver, f"{r.points:.0f} pts")
        cols[i].caption("🥇 Leading" if behind == 0 else f"{behind:.0f} points behind")
    if len(alive) > 4:
        st.caption(f"…and {len(alive) - 4} more still mathematically in it.")
    out = len(standings) - len(alive)
    if out:
        st.caption(f"{out} drivers can no longer win the title.")

# ------------------------------------------------------------- the race for the title
st.header("The title fight, race by race")
st.caption("Press ▶ to watch the standings change after every round.")
top = standings.head(10).driverRef.tolist()
rounds = sorted(res["round"].unique())
per_round = (pts[pts.driverRef.isin(top)].groupby(["round", "driverRef"]).points.sum()
             .unstack(fill_value=0).reindex(rounds, fill_value=0).cumsum())
frames = per_round.stack().rename("points").reset_index()
frames["rank"] = frames.groupby("round").points.rank(ascending=False, method="first")
info = standings.set_index("driverRef")
frames = frames.join(info[["driver", "code", "constructorRef"]], on="driverRef")
names = schedule.set_index("round")["name"].str.replace(" Grand Prix", " GP")
frames["after"] = frames["round"].map(lambda r: f"After round {r}: {names.get(r, '')}")
fig = px.bar(frames, x="points", y="rank", orientation="h", text="driver", color="constructorRef",
             color_discrete_map={r: team_color(r) for r in frames.constructorRef.unique()},
             animation_frame="after", range_x=[0, frames.points.max() * 1.15],
             labels={"points": "Points", "rank": ""})
fig.update_traces(textposition="inside", insidetextanchor="start", textfont_color="white")
fig.update_yaxes(autorange="reversed", tickvals=list(range(1, 11)), ticktext=[f"P{i}" for i in range(1, 11)])
fig.update_layout(showlegend=False)
for k in ("frame", "transition"):
    fig.layout.updatemenus[0].buttons[0].args[1][k]["duration"] = 700
fig.layout.sliders[0].active = len(rounds) - 1
fig.layout.sliders[0].currentvalue.prefix = ""
for trace, final in zip(fig.data, fig.frames[-1].data):
    trace.update(x=final.x, y=final.y, text=final.text)
st.plotly_chart(style(fig, 520), width="stretch")

# ------------------------------------------------------------- teams
st.header("Teams")
teams = pts.groupby(["constructorRef", "team"]).points.sum().reset_index().drop_duplicates("constructorRef", keep="last")
teams = teams.sort_values("points")
fig = px.bar(teams, x="points", y="team", orientation="h", color="constructorRef", text="points",
             color_discrete_map={r: team_color(r) for r in teams.constructorRef}, labels={"points": "Points", "team": ""})
fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
fig.update_layout(showlegend=False)
st.plotly_chart(style(fig, 420), width="stretch")
st.caption("Each team has two drivers; their points add up for the teams' championship.")
