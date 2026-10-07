import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1app.charts import style, team_color
from views.common import get_data

data = get_data()
res, q = data.results, data.qualifying
st.title("⚔️ Driver head-to-head")

# Drivers ordered by most recent race first, so current drivers come first.
latest = res.sort_values("date").drop_duplicates("driverRef", keep="last").sort_values("date", ascending=False)
names = dict(zip(latest.driverRef, latest.driver + " (" + latest.year.astype(str) + ")"))
refs = list(names)
current = latest[latest.year == latest.year.max()]
# Default to the leading team's pairing in the latest season.
lead_team = res[res.year == latest.year.max()].groupby("constructorRef").points.sum().idxmax()
pair = current[current.constructorRef == lead_team].driverRef.tolist()[:2] or refs[:2]

a, b, c = st.columns([2, 2, 2])
d1 = a.selectbox("Driver 1", refs, index=refs.index(pair[0]), format_func=names.get)
d2 = b.selectbox("Driver 2", refs, index=refs.index(pair[-1] if len(pair) > 1 else refs[1]), format_func=names.get)
if d1 == d2:
    st.warning("Pick two different drivers.")
    st.stop()
years = sorted(set(res[res.driverRef == d1].year) | set(res[res.driverRef == d2].year), reverse=True)
scope = c.selectbox("Period", ["Whole career"] + years)

r = res[res.driverRef.isin([d1, d2])]
qq = q[q.driverRef.isin([d1, d2])]
if scope != "Whole career":
    r, qq = r[r.year == scope], qq[qq.year == scope]
label = {d1: names[d1].split(" (")[0], d2: names[d2].split(" (")[0]}
col = {d1: team_color(r[r.driverRef == d1].constructorRef.iloc[-1]) if (r.driverRef == d1).any() else "#e10600",
       d2: team_color(r[r.driverRef == d2].constructorRef.iloc[-1]) if (r.driverRef == d2).any() else "#888"}
if col[d1] == col[d2]:
    col[d2] = "#AAAAAA"

# ------------------------------------------------------------- summary
def summary(df: pd.DataFrame, qdf: pd.DataFrame, d: str) -> dict:
    x, y = df[df.driverRef == d], qdf[qdf.driverRef == d]
    return {"Races": len(x), "Wins": int((x.positionOrder == 1).sum()), "Podiums": int((x.positionOrder <= 3).sum()),
            "Points": x.points.sum(), "Avg finish": x.positionOrder.mean(), "Avg quali": y.position.mean(),
            "Poles": int((x.grid == 1).sum()), "DNFs": int((~x.finished.astype(bool)).sum())}

st.dataframe(pd.DataFrame({label[d1]: summary(r, qq, d1), label[d2]: summary(r, qq, d2)}).T,
             width="stretch",
             column_config={"Avg finish": st.column_config.NumberColumn(format="%.2f"),
                            "Avg quali": st.column_config.NumberColumn(format="%.2f"),
                            "Points": st.column_config.NumberColumn(format="%.0f")})

# ------------------------------------------------------------- direct battles
both = r.pivot_table(index="raceId", columns="driverRef", values="positionOrder", aggfunc="first").dropna()
fin = r[r.finished.astype(bool)].pivot_table(index="raceId", columns="driverRef", values="positionOrder").dropna()
qb = qq.pivot_table(index="raceId", columns="driverRef", values="position").dropna()

st.subheader("When they raced each other")
if both.empty:
    st.info("These two never started the same race in this period.")
    st.stop()

def h2h(df):
    return (int((df[d1] < df[d2]).sum()), int((df[d2] < df[d1]).sum())) if len(df) else (0, 0)

rows = [("Qualifying", *h2h(qb)), ("Race (both finished)", *h2h(fin)), ("Race (all)", *h2h(both))]
fig = go.Figure()
for name, w1, w2 in rows:
    total = max(w1 + w2, 1)
    fig.add_trace(go.Bar(y=[name], x=[w1 / total], orientation="h", marker_color=col[d1], text=f"{label[d1]} {w1}",
                         textposition="inside", showlegend=False, hovertemplate=f"{label[d1]}: {w1}<extra></extra>"))
    fig.add_trace(go.Bar(y=[name], x=[w2 / total], orientation="h", marker_color=col[d2], text=f"{w2} {label[d2]}",
                         textposition="inside", insidetextanchor="end", showlegend=False,
                         hovertemplate=f"{label[d2]}: {w2}<extra></extra>"))
fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]))
fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
st.plotly_chart(style(fig, 260), width="stretch")

# ------------------------------------------------------------- timeline
st.subheader("Finishing positions over time")
shared = r[r.raceId.isin(both.index)].sort_values("date")
shared["race"] = shared.year.astype(str) + " " + shared.name.str.replace(" Grand Prix", " GP")
shared["who"] = shared.driverRef.map(label)
fig = px.line(shared, x="date", y="positionOrder", color="who", markers=True, hover_data=["race", "team", "status"],
              color_discrete_map={label[d]: col[d] for d in (d1, d2)}, labels={"positionOrder": "Finish", "date": "", "who": ""})
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style(fig, 420), width="stretch")

# ------------------------------------------------------------- by circuit
st.subheader("Average finish by circuit")
by_c = (shared.groupby(["circuitRef", "who"]).positionOrder.mean().unstack().dropna()
        .assign(diff=lambda x: x[label[d1]] - x[label[d2]]).sort_values("diff"))
if by_c.empty:
    st.write("Not enough shared races per circuit.")
else:
    by_c = by_c.reset_index()
    by_c["better"] = by_c["diff"].map(lambda v: label[d1] if v < 0 else label[d2])
    fig = px.bar(by_c, x="diff", y="circuitRef", orientation="h", color="better",
                 color_discrete_map={label[d]: col[d] for d in (d1, d2)},
                 labels={"diff": f"← {label[d1]} better · {label[d2]} better →", "circuitRef": ""})
    st.plotly_chart(style(fig, max(300, 22 * len(by_c))), width="stretch")
