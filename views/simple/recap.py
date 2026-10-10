import plotly.express as px
import streamlit as st

from f1app.charts import style, team_color
from views.common import get_data
from views.simple.widgets import ordinal, podium, watch_replay_button

data = get_data()
done = data.completed_races()
labels = {r.raceId: f"{r.year} · {r.name}" for r in done.iloc[::-1].head(60).itertuples()}
race_id = st.selectbox("Race", list(labels), format_func=labels.get, label_visibility="collapsed")
race = done.set_index("raceId").loc[race_id]
res = data.results[data.results.raceId == race_id].sort_values("positionOrder")

st.title(f"📰 {race['name']} {race.year}")
st.caption(f"{race.circuit} · {race.date:%A %d %B %Y}")

# ------------------------------------------------------------- the story
w, p2, p3 = res.iloc[0], res.iloc[1], res.iloc[2]
started = int(w.grid) if w.grid and w.grid > 0 else None
where = ("from pole position" if started == 1 else f"after starting {ordinal(started)}" if started else "from the pit lane")
story = [f"🏆 **{w.driver}** won for **{w.team}** {where}, ahead of {p2.driver} and {p3.driver}."]

movers = res[(res.grid > 0) & res.finished.astype(bool)].assign(gain=lambda x: x.grid - x.positionOrder)
if len(movers) and movers.gain.max() >= 3:
    m = movers.loc[movers.gain.idxmax()]
    story.append(f"🚀 Biggest climber: **{m.driver}** went from {ordinal(m.grid)} on the grid to "
                 f"{ordinal(m.positionOrder)} (+{int(m.gain)} places).")
if len(movers) and movers.gain.min() <= -3:
    m = movers.loc[movers.gain.idxmin()]
    story.append(f"📉 Tough day for **{m.driver}**: started {ordinal(m.grid)}, finished {ordinal(m.positionOrder)}.")

dnf = res[~res.finished.astype(bool)]
if len(dnf):
    names = ", ".join(f"{r.driver} ({r.status.lower()})" for r in dnf.itertuples())
    story.append(f"💥 {len(dnf)} driver{'s' if len(dnf) > 1 else ''} didn't finish: {names}.")
else:
    story.append("✅ Every car made it to the finish.")

# Championship before vs after.
season = data.results[data.results.year == race.year]
spr = data.sprints[data.sprints.year == race.year]
def table(upto_round):
    pts = (season[season["round"] <= upto_round].groupby("driver").points.sum()
           .add(spr[spr["round"] <= upto_round].groupby("driver").points.sum(), fill_value=0))
    return pts.sort_values(ascending=False)
after, before = table(race["round"]), table(race["round"] - 1)
leader = after.index[0]
gap = after.iloc[0] - after.iloc[1]
if len(before) and before.index[0] != leader:
    story.append(f"👑 **{leader}** takes the championship lead from {before.index[0]}! "
                 f"Now {gap:.0f} points ahead of {after.index[1]}.")
else:
    story.append(f"👑 **{leader}** leads the championship by **{gap:.0f} points** over {after.index[1]}.")

for line in story:
    st.markdown(line)

podium([{"code": r.code, "name": r.driver, "team": r.team, "ref": r.constructorRef,
         "line": f"+{r.points:.0f} points"} for r in res.head(3).itertuples()])

# ------------------------------------------------------------- places gained
st.subheader("Who moved forward, who went backwards?")
st.caption("Starting position compared with where they finished. Green = gained places.")
mv = movers.sort_values("gain")
mv["label"] = mv.driver + "  (P" + mv.grid.astype(int).astype(str) + " → P" + mv.positionOrder.astype(str) + ")"
fig = px.bar(mv, x="gain", y="label", orientation="h", color=mv.gain > 0,
             color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
             labels={"gain": "Places gained", "label": ""})
fig.update_layout(showlegend=False)
st.plotly_chart(style(fig, max(320, 26 * len(mv))), width="stretch")

# ------------------------------------------------------------- full result
st.subheader("Full result")
rows = []
for r in res.itertuples():
    color = team_color(r.constructorRef)
    status = f"{r.points:.0f} pts" if r.points else ("" if r.finished else "❌ " + r.status)
    rows.append(f"<tr><td style='text-align:right;font-weight:700'>{r.positionText}</td>"
                f"<td><span style='display:inline-block;width:5px;height:16px;background:{color};"
                f"border-radius:2px;vertical-align:middle'></span></td>"
                f"<td><b>{r.driver}</b></td><td style='opacity:.7'>{r.team}</td><td>{status}</td></tr>")
st.html("<table style='width:100%;border-collapse:collapse;font-size:15px'>"
        "<style>td{padding:5px 8px;border-bottom:1px solid rgba(128,128,128,.2)}</style>"
        + "".join(rows) + "</table>")

if race.year >= 2018:
    watch_replay_button(race_id, "Watch this race in the replay")
