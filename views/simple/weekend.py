import plotly.express as px
import streamlit as st

from f1app import sessions
from f1app.charts import style, team_color
from views.common import get_data, get_prediction, get_track_map
from views.simple.widgets import countdown, odds_in_words, podium, track_figure, watch_replay_button

data = get_data()
upcoming = data.upcoming_races()
if upcoming.empty:
    st.title("🏁 Season complete")
    st.write("There are no more races this year. Check the **Championship** page for the final standings.")
    st.stop()

race = upcoming.iloc[0]
sim, meta = get_prediction(int(race.raceId))

st.title(f"🏁 Next up: {race['name']}")
st.markdown(f"**{race.circuit}** · {race.location}, {race.country} · Round {race['round']} of "
            f"{int(data.races.query('year == @race.year')['round'].max())}"
            + (" · 🏃 sprint weekend (an extra short race on Saturday)" if race.has_sprint else ""))
countdown(race.start_utc.isoformat(), "Lights out:")

# ------------------------------------------------------------- prediction
st.header("Who's going to win?")
top3 = sim.sort_values("p_win", ascending=False).head(3)
podium([{"code": r.code, "name": r.driver, "team": r.team, "ref": r.constructorRef,
         "line": f"{r.p_win:.0%} chance to win"} for r in top3.itertuples()],
       caption="Our three favourites, based on 20,000 simulated races. "
               + ("Uses the real starting grid." if meta["uses_grid"]
                  else "Qualifying hasn't happened yet, so this will sharpen on Saturday."))

fav = top3.iloc[0]
st.info(f"**{fav.driver}** is the favourite, but even the favourite only wins **{odds_in_words(fav.p_win)}** times. "
        f"F1 races are unpredictable: there's a **{1 - top3.p_win.sum():.0%}** chance someone else outside our top 3 wins.")

# ------------------------------------------------------------- pick a driver
st.header("How will your driver do?")
names = dict(zip(sim.driverRef, sim.driver + "  ·  " + sim.team))
pick = st.selectbox("Pick a driver", list(names), format_func=names.get, index=0)
d = sim.set_index("driverRef").loc[pick]
recent = (data.results[data.results.driverRef == pick].sort_values("date").tail(5))

left, right = st.columns([3, 2])
with left:
    c1, c2, c3 = st.columns(3)
    for col, label, p in ((c1, "Win", d.p_win), (c2, "Podium (top 3)", d.p_podium), (c3, "Points (top 10)", d.p_points)):
        col.metric(label, f"{p:.0%}")
        col.caption(odds_in_words(p).capitalize())
    lo, hi = round(d.fin_p10), round(d.fin_p90)
    first = d.driver.split()[0]
    st.markdown(f"Most likely result: **P{int(d.predicted_pos)}**. When {first} makes it to the finish, "
                f"8 out of 10 simulated races end between **P{lo}** and **P{hi}**. "
                f"Chance of not finishing at all: **{d.p_dnf:.0%}** ({odds_in_words(d.p_dnf)}).")
with right:
    if len(recent):
        recent = recent.assign(label=recent.name.str.replace(" Grand Prix", ""))
        fig = px.bar(recent, x="label", y=recent.positionOrder.rsub(23), text="positionOrder",
                     color_discrete_sequence=[team_color(d.constructorRef)],
                     labels={"label": "", "y": ""})
        fig.update_traces(texttemplate="P%{text}", textposition="outside",
                          hovertemplate="%{x}: P%{text}<extra></extra>")
        fig.update_yaxes(visible=False, range=[0, 26])
        fig.update_layout(title=dict(text="Last 5 races (taller = better)", font=dict(size=14)))
        st.plotly_chart(style(fig, 260), width="stretch")

# ------------------------------------------------------------- everyone's chances
with st.expander("See everyone's chances"):
    ranked = sim.sort_values("p_podium")
    fig = px.bar(ranked, x="p_podium", y="driver", orientation="h", color="constructorRef",
                 color_discrete_map={r: team_color(r) for r in sim.constructorRef},
                 labels={"p_podium": "Chance of a podium", "driver": ""}, hover_data={"constructorRef": False, "team": True})
    fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
    st.plotly_chart(style(fig, 640), width="stretch")

# ------------------------------------------------------------- the track
st.header("The track")
past = data.completed_races().query("circuitRef == @race.circuitRef and year >= @sessions.FASTF1_FROM")
left, right = st.columns([3, 2])
with left:
    if past.empty:
        st.write("A new track for F1 — no map yet. It'll appear after the first session.")
    else:
        last = past.iloc[-1]
        try:
            st.plotly_chart(track_figure(get_track_map(int(last.year), int(last["round"]))), width="stretch")
            st.caption(f"Track layout from the {last.year} race. Numbers are the corners.")
        except Exception:
            st.write("Track map unavailable right now.")
with right:
    here = data.results[(data.results.circuitRef == race.circuitRef)]
    winners = here[here.positionOrder == 1].sort_values("date", ascending=False)
    if winners.empty:
        st.write("First time F1 races here!")
    else:
        w = winners.iloc[0]
        st.markdown(f"**Last winner here:** {w.driver} ({w.team}, {w.year})")
        champs = (winners[winners.driverRef.isin(sim.driverRef)].driver.value_counts().head(3))
        if len(champs):
            st.markdown("**Most wins here among today's drivers:**")
            for name, n in champs.items():
                st.markdown(f"- {name}: {n} win{'s' if n > 1 else ''}")
        poles = here[here.grid == 1]
        if len(poles):
            st.markdown(f"**Does pole matter here?** The driver starting first has won "
                        f"**{poles.positionOrder.eq(1).mean():.0%}** of {len(poles)} races at this track.")

watch_replay_button(int(data.completed_races().raceId.iloc[-1]), "Watch the last race again in the replay")
