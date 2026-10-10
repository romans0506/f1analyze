import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from f1app import sessions
from f1app.charts import COMPOUND_COLORS, driver_colors, driver_dashes, style, team_color
from f1app.data import load_lap_positions
from views.common import get_data, get_session, race_picker, season_picker

data = get_data()
year = season_picker(data, key="gp_year")
race = race_picker(data, year, key="gp_race")
race_id = int(race.name)

st.title(f"🏎️ {race['name']} {year}")
st.caption(f"Round {race['round']} · {race.circuit}, {race.location} · {race.date:%d %B %Y}")

res = data.results[data.results.raceId == race_id].sort_values("positionOrder").copy()
res["gained"] = (res.grid.where(res.grid > 0) - res.positionOrder)
code_colors = driver_colors(res, "code")
code_dashes = driver_dashes(res, "code")
winner = res.iloc[0]
pole = res[res.grid == 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Winner", winner.driver.split()[-1], winner.team, delta_color="off")
c2.metric("Pole", pole.iloc[0].driver.split()[-1] if len(pole) else "—",
          f"finished P{pole.iloc[0].positionOrder}" if len(pole) else None, delta_color="off")
best = res.loc[res.gained.idxmax()] if res.gained.notna().any() else None
c3.metric("Biggest climber", best.driver.split()[-1] if best is not None else "—",
          f"+{best.gained:.0f} places" if best is not None else None)
c4.metric("Finishers", f"{int(res.finished.sum())} / {len(res)}")

# ------------------------------------------------------------- results
tab_res, tab_pos, tab_pace, tab_tyre, tab_deg, tab_gap, tab_tel, tab_q = st.tabs(
    ["Results", "Positions", "Race pace", "Tyre strategy", "Tyre degradation", "Gaps", "Telemetry", "Qualifying"])

with tab_res:
    left, right = st.columns([3, 2])
    with left:
        table = res[["positionText", "driver", "team", "grid", "gained", "points", "laps", "status", "fastestLapTime"]]
        st.dataframe(table.rename(columns={"positionText": "Pos", "driver": "Driver", "team": "Team", "grid": "Grid",
                                           "gained": "+/-", "points": "Pts", "laps": "Laps", "status": "Status",
                                           "fastestLapTime": "Fastest lap"}),
                     hide_index=True, width="stretch", height=600,
                     column_config={"+/-": st.column_config.NumberColumn(format="%+d"),
                                    "Grid": st.column_config.NumberColumn(format="%d")})
    with right:
        st.markdown("**Grid → finish**")
        fig = go.Figure()
        for r in res[res.grid > 0].itertuples():
            fig.add_trace(go.Scatter(x=[r.grid, r.positionOrder], y=[r.code, r.code], mode="lines+markers",
                                     line=dict(color=code_colors[r.code], width=3),
                                     marker=dict(size=[6, 12], symbol=["circle-open", "circle"]),
                                     hovertemplate=f"{r.driver}: P{r.grid:.0f} → P{r.positionOrder}<extra></extra>",
                                     showlegend=False))
        fig.update_yaxes(categoryorder="array", categoryarray=res.code.tolist()[::-1])
        fig.update_xaxes(title="Position (hollow = grid, filled = finish)", autorange="reversed", dtick=2)
        st.plotly_chart(style(fig, 600), width="stretch")

has_timing = year >= sessions.FASTF1_FROM
session = None
if has_timing:
    try:
        session = get_session(year, int(race["round"]), "R")
    except Exception as exc:  # FastF1 data can be missing for a few events
        st.warning(f"Detailed timing isn't available for this race ({exc}).")

if session is not None:
    laps = sessions.lap_table(session)
    abbr_team = res.set_index("code").constructorRef.to_dict()
    laps["constructorRef"] = laps.Driver.map(abbr_team)
    colors = {d: team_color(abbr_team.get(d, "")) for d in laps.Driver.unique()}
    dashes = {d: code_dashes.get(d, "solid") for d in laps.Driver.unique()}
    order = [c for c in res.code if c in colors]

with tab_pos:
    if session is not None:
        pos = laps.dropna(subset=["Position"])
        fig = px.line(pos, x="LapNumber", y="Position", color="Driver", line_dash="Driver",
                      color_discrete_map=colors, line_dash_map=dashes, category_orders={"Driver": order},
                      labels={"LapNumber": "Lap"})
    else:
        lp = st.cache_data(load_lap_positions)(race_id)
        if lp.empty:
            st.info("No lap-by-lap data for this race (the archive starts in 1996).")
            fig = None
        else:
            fig = px.line(lp, x="lap", y="position", color="code", color_discrete_map=code_colors,
                          line_dash="code", line_dash_map=code_dashes, labels={"lap": "Lap", "position": "Position"})
    if fig is not None:
        fig.update_yaxes(autorange="reversed", dtick=1)
        st.plotly_chart(style(fig, 640), width="stretch")

needs_timing = "Lap-time detail comes from the official timing archive, which starts in 2018."

with tab_pace:
    if session is None:
        st.info(needs_timing)
    else:
        clean = sessions.race_pace(laps)
        st.caption("Green-flag racing laps only (no pit laps, safety cars or outliers). Lower and tighter = faster and more consistent.")
        fig = px.box(clean, x="Driver", y="LapSeconds", color="Driver", color_discrete_map=colors,
                     category_orders={"Driver": order}, points="outliers", labels={"LapSeconds": "Lap time (s)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 520), width="stretch")

        st.markdown("**Lap times through the race**")
        picks = st.multiselect("Drivers", order, default=order[:4], key="pace_drivers")
        sub = clean[clean.Driver.isin(picks)]
        fig = px.scatter(sub, x="LapNumber", y="LapSeconds", color="Driver", symbol="Compound",
                         color_discrete_map=colors,
                         labels={"LapNumber": "Lap", "LapSeconds": "Lap time (s)"})
        st.plotly_chart(style(fig, 480), width="stretch")

with tab_tyre:
    if session is None:
        st.info(needs_timing)
    else:
        stints = sessions.stints(laps)
        fig = go.Figure()
        for s in stints.itertuples():
            fig.add_trace(go.Bar(y=[s.Driver], x=[s.end - s.start + 1], base=[s.start - 1], orientation="h",
                                 marker=dict(color=COMPOUND_COLORS.get(s.Compound, "#888"), line=dict(color="#222", width=1)),
                                 name=s.Compound, legendgroup=s.Compound, showlegend=False,
                                 hovertemplate=f"{s.Driver} · stint {s.Stint:.0f}<br>{s.Compound}: laps {s.start:.0f}–{s.end:.0f}<extra></extra>"))
        for comp in stints.Compound.unique():
            fig.add_trace(go.Bar(y=[None], x=[None], marker_color=COMPOUND_COLORS.get(comp, "#888"), name=comp))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
        fig.update_xaxes(title="Lap")
        st.plotly_chart(style(fig, 640), width="stretch")

        stops = stints.groupby("Driver").Stint.max().sub(1).reindex(order).rename("Pit stops")
        st.markdown("**Stops per driver**")
        st.bar_chart(stops)

with tab_deg:
    if session is None:
        st.info(needs_timing)
    else:
        fuel = st.slider("Fuel correction (s gained per lap as fuel burns)", 0.0, 0.08, sessions.FUEL_EFFECT, 0.005,
                         help="Cars get lighter, and so faster, as they burn fuel. Adding that back isolates tyre wear.")
        deg = sessions.degradation(sessions.race_pace(laps), fuel)
        if deg["laps"].empty:
            st.info("Not enough clean laps to measure degradation.")
        else:
            st.caption("Fuel-corrected lap time against tyre age. The slope of each line is how much time that "
                       "compound lost per lap of wear (positive = slower as the tyre ages).")
            dl = deg["laps"]
            fig = px.scatter(dl, x="TyreLife", y="Corrected", color="Compound", color_discrete_map=COMPOUND_COLORS,
                             opacity=0.35, hover_data=["Driver", "LapNumber"],
                             labels={"TyreLife": "Tyre age (laps)", "Corrected": "Fuel-corrected lap time (s)"})
            for comp, fit in deg["compound_fit"].items():
                fig.add_scatter(x=fit["x"], y=fit["y"], mode="lines", name=f"{comp} trend ({fit['slope']:+.3f} s/lap)",
                                line=dict(color=COMPOUND_COLORS.get(comp, "#888"), width=4))
            st.plotly_chart(style(fig, 480), width="stretch")

            st.markdown("**Degradation per stint** (stints of 6+ clean laps)")
            if deg["stints"].empty:
                st.info("No stint had enough clean laps (common in wet or safety-car-heavy races).")
            else:
                stints_deg = deg["stints"].sort_values("deg_per_lap")
                fig = px.bar(stints_deg, x="deg_per_lap", y="label", orientation="h", color="Compound",
                             color_discrete_map=COMPOUND_COLORS, hover_data=["laps", "avg_pace"],
                             labels={"deg_per_lap": "Seconds lost per lap of tyre age", "label": ""})
                fig.update_yaxes(categoryorder="array", categoryarray=stints_deg.label.tolist())
                st.plotly_chart(style(fig, max(320, 22 * len(stints_deg))), width="stretch")

with tab_gap:
    if session is None:
        st.info(needs_timing)
    else:
        t = session.laps[["Driver", "LapNumber", "Time"]].dropna()
        t["Time"] = t.Time.dt.total_seconds()
        leader = t.groupby("LapNumber").Time.min()
        t["Gap"] = t.Time - t.LapNumber.map(leader)
        picks = st.multiselect("Drivers", order, default=order[:6], key="gap_drivers")
        fig = px.line(t[t.Driver.isin(picks)], x="LapNumber", y="Gap", color="Driver", line_dash="Driver",
                      color_discrete_map=colors, line_dash_map=dashes,
                      labels={"LapNumber": "Lap", "Gap": "Gap to leader (s)"})
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 520), width="stretch")

with tab_tel:
    if not has_timing:
        st.info(needs_timing)
    else:
        kind = st.radio("Session", ["Qualifying", "Race"], horizontal=True)
        a, b = st.columns(2)
        d1 = a.selectbox("Driver 1", res.code.tolist(), index=0)
        d2 = b.selectbox("Driver 2", res.code.tolist(), index=1)
        if st.button("Compare fastest laps", type="primary"):
            try:
                tel_session = get_session(year, int(race["round"]), "Q" if kind == "Qualifying" else "R", telemetry=True)
                tel = pd.concat([sessions.fastest_lap_telemetry(tel_session, d) for d in (d1, d2)])
            except Exception as exc:
                st.error(f"Couldn't load telemetry: {exc}")
                tel = pd.DataFrame()
            if tel.empty:
                st.warning("No telemetry for that selection.")
            else:
                lt = tel.groupby("Driver").LapTime.first()
                st.markdown(f"**{d1}** {lt.get(d1, float('nan')):.3f}s vs **{d2}** {lt.get(d2, float('nan')):.3f}s "
                            f"(Δ {abs(lt.get(d1, 0) - lt.get(d2, 0)):.3f}s)")
                c = {d1: code_colors.get(d1, "#e10600"),
                     d2: code_colors.get(d2, "#888") if code_colors.get(d2) != code_colors.get(d1) else "#AAAAAA"}
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                    row_heights=[0.45, 0.2, 0.15, 0.2],
                                    subplot_titles=("Speed (km/h)", "Throttle (%)", "Brake", "Gear"))
                for d in (d1, d2):
                    s = tel[tel.Driver == d]
                    for row, col in enumerate(["Speed", "Throttle", "Brake", "nGear"], start=1):
                        fig.add_trace(go.Scatter(x=s.Distance, y=s[col].astype(float), name=d, legendgroup=d,
                                                 showlegend=row == 1, line=dict(color=c[d], width=1.6)), row=row, col=1)
                fig.update_xaxes(title="Distance (m)", row=4, col=1)
                st.plotly_chart(style(fig, 760), width="stretch")

with tab_q:
    q = data.qualifying[data.qualifying.raceId == race_id].sort_values("position")
    if q.empty:
        st.info("No qualifying data for this race.")
    else:
        def secs(t):
            try:
                m, s = str(t).split(":")
                return int(m) * 60 + float(s)
            except ValueError:
                return None
        q = q.assign(best=q[["q3", "q2", "q1"]].bfill(axis=1).iloc[:, 0])
        q["best_s"] = q.best.map(secs)
        q["gap"] = q.best_s - q.best_s.min()
        left, right = st.columns([3, 2])
        left.dataframe(q[["position", "driver", "team", "q1", "q2", "q3"]]
                       .rename(columns={"position": "Pos", "driver": "Driver", "team": "Team",
                                        "q1": "Q1", "q2": "Q2", "q3": "Q3"}),
                       hide_index=True, width="stretch", height=600)
        fig = px.bar(q.iloc[::-1], x="gap", y="code", orientation="h", color="code",
                     color_discrete_map=driver_colors(q, "code"),
                     labels={"gap": "Gap to pole, best lap (s)", "code": ""})
        fig.update_layout(showlegend=False)
        right.plotly_chart(style(fig, 600), width="stretch")
