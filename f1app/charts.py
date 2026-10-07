"""Shared colours and Plotly helpers."""
from __future__ import annotations

import plotly.express as px

TEAM_COLORS = {
    "mercedes": "#27F4D2", "red_bull": "#3671C6", "ferrari": "#E8002D", "mclaren": "#FF8000",
    "aston_martin": "#229971", "alpine": "#FF87BC", "williams": "#64C4FF", "rb": "#6692FF",
    "alphatauri": "#5E8FAA", "toro_rosso": "#469BFF", "sauber": "#52E252", "audi": "#BB0A30",
    "alfa": "#C92D4B", "haas": "#B6BABD", "cadillac": "#C8A951", "racing_point": "#F596C8",
    "force_india": "#F596C8", "renault": "#FFF500", "lotus_f1": "#FFB800",
}
COMPOUND_COLORS = {
    "SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC", "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD", "UNKNOWN": "#888888", "TEST_UNKNOWN": "#888888",
}
_FALLBACK = px.colors.qualitative.Bold


def team_color(ref: str) -> str:
    if ref in TEAM_COLORS:
        return TEAM_COLORS[ref]
    return _FALLBACK[hash(ref) % len(_FALLBACK)]


def driver_colors(df, driver_col: str, team_col: str = "constructorRef") -> dict:
    """Driver -> team colour. Second driver of a team gets the same colour;
    use ``driver_dashes`` to tell teammates apart."""
    pairs = df[[driver_col, team_col]].drop_duplicates(driver_col, keep="last")
    return {d: team_color(t) for d, t in zip(pairs[driver_col], pairs[team_col])}


def driver_dashes(df, driver_col: str, team_col: str = "constructorRef") -> dict:
    pairs = df[[driver_col, team_col]].drop_duplicates(driver_col, keep="last")
    seen, out = set(), {}
    for d, t in zip(pairs[driver_col], pairs[team_col]):
        out[d] = "dash" if t in seen else "solid"
        seen.add(t)
    return out


def style(fig, height: int = 480):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="closest")
    return fig
