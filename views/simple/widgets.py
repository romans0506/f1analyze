"""Friendly building blocks for Simple mode."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from f1app.charts import team_color


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def watch_replay_button(race_id: int, label: str) -> None:
    """Open the replay page on a specific race."""
    if st.button(label, icon="📺", type="primary"):
        st.session_state["replay_race"] = int(race_id)
        st.switch_page("views/replay.py")


def odds_in_words(p: float) -> str:
    """0.27 -> 'about 1 in 4'."""
    if p >= 0.95:
        return "almost certain"
    if p >= 0.5:
        return "more likely than not"
    if p < 0.005:
        return "very unlikely"
    return f"about 1 in {max(2, round(1 / p))}"


def podium(entries: list[dict], caption: str = "") -> None:
    """``entries`` = [{code, name, team, ref, line}] for P1, P2, P3."""
    heights = {1: 150, 2: 110, 3: 80}
    blocks = []
    for place in (2, 1, 3):  # classic podium layout
        if place > len(entries):
            continue
        e = entries[place - 1]
        color = team_color(e["ref"])
        blocks.append(f"""
        <div class="spot">
          <div class="who"><div class="code">{html.escape(e['code'])}</div>
            <div class="name">{html.escape(e['name'])}</div>
            <div class="team">{html.escape(e['team'])}</div>
            <div class="line">{html.escape(e.get('line', ''))}</div></div>
          <div class="step" style="height:{heights[place]}px;background:{color}"><span>{place}</span></div>
        </div>""")
    st.html(f"""
    <style>
      .podium {{ display:flex; justify-content:center; align-items:flex-end; gap:10px; margin:6px 0 4px; }}
      .podium .spot {{ flex:1; max-width:210px; text-align:center; }}
      .podium .code {{ font-size:28px; font-weight:800; letter-spacing:1px; }}
      .podium .name {{ font-weight:600; }}
      .podium .team, .podium .line {{ font-size:13px; opacity:.75; }}
      .podium .line {{ margin-bottom:6px; font-weight:600; opacity:.9; }}
      .podium .step {{ border-radius:10px 10px 0 0; display:flex; align-items:flex-start; justify-content:center;
                      box-shadow: inset 0 -30px 40px rgba(0,0,0,.18); }}
      .podium .step span {{ font-size:40px; font-weight:900; color:rgba(255,255,255,.92);
                           text-shadow:0 2px 6px rgba(0,0,0,.35); margin-top:6px; }}
      .podium-caption {{ text-align:center; font-size:13px; opacity:.7; }}
    </style>
    <div class="podium">{''.join(blocks)}</div>
    <div class="podium-caption">{html.escape(caption)}</div>
    """)


def countdown(target_iso: str, label: str) -> None:
    """A ticking countdown to ``target_iso`` (UTC), in the viewer's local time."""
    st.iframe(f"""
    <div style="font-family:system-ui,sans-serif;display:flex;gap:14px;align-items:center;flex-wrap:wrap;color:#888">
      <div id="cd" style="display:flex;gap:8px"></div>
      <div id="when" style="font-size:14px"></div>
    </div>
    <script>
      const target = new Date("{target_iso}");
      const fmt = new Intl.DateTimeFormat(undefined, {{weekday:"long", day:"numeric", month:"long", hour:"2-digit", minute:"2-digit"}});
      document.getElementById("when").textContent = "{html.escape(label)} " + fmt.format(target) + " (your time)";
      const box = (v, u) => `<div style="background:#e10600;color:#fff;border-radius:10px;padding:6px 12px;text-align:center;min-width:62px">
          <div style="font-size:26px;font-weight:800;line-height:1">${{v}}</div><div style="font-size:11px;letter-spacing:1px">${{u}}</div></div>`;
      function tick() {{
        let s = Math.max(0, Math.floor((target - new Date()) / 1000));
        const d = Math.floor(s / 86400); s %= 86400; const h = Math.floor(s / 3600); s %= 3600;
        const m = Math.floor(s / 60); s %= 60;
        document.getElementById("cd").innerHTML = box(d, "DAYS") + box(h, "HRS") + box(m, "MIN") + box(s, "SEC");
      }}
      tick(); setInterval(tick, 1000);
    </script>""", height=80)


def track_figure(track: dict, color: str = "#e10600", height: int = 380) -> go.Figure:
    xs, ys = zip(*track["track"])
    xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(128,128,128,.35)", width=16),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode="markers+text", text=["Start / finish"],
                             textposition="top center", marker=dict(size=12, color="white", line=dict(color="black", width=2)),
                             hoverinfo="skip"))
    if track["corners"]:
        fig.add_trace(go.Scatter(x=[c["x"] for c in track["corners"]], y=[c["y"] for c in track["corners"]],
                                 mode="markers+text", text=[c["n"] for c in track["corners"]],
                                 textfont=dict(size=10, color="white"),
                                 marker=dict(size=18, color="rgba(40,40,55,.85)"),
                                 hovertemplate="Turn %{text}<extra></extra>"))
    fig.update_layout(showlegend=False, height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
