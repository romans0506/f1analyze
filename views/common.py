"""Cached loaders shared by every page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1app.data import F1Data, load_data
from f1app import model, sessions


@st.cache_resource(ttl="6h", show_spinner="Loading results database…")
def get_data() -> F1Data:
    return load_data()


@st.cache_resource(max_entries=6, show_spinner="Downloading session timing data (first load can take a minute)…")
def get_session(year: int, rnd: int, kind: str, telemetry: bool = False):
    return sessions.load_session(year, rnd, kind, telemetry=telemetry)


@st.cache_data(ttl="6h", show_spinner="Simulating the race 20,000 times…")
def get_prediction(race_id: int) -> tuple[pd.DataFrame, dict]:
    sim, m = model.predict_race(get_data(), race_id)
    return sim, {"uses_grid": m.uses_grid, "dnf_prior": m.dnf_prior, "features": m.features}


@st.cache_data(ttl="24h", show_spinner="Back-testing the model race by race (≈1 min)…")
def get_backtest(seasons: tuple[int, ...], use_grid: bool) -> pd.DataFrame:
    return model.backtest(get_data(), list(seasons), use_grid=use_grid)


def season_picker(data: F1Data, key: str, label: str = "Season") -> int:
    return st.sidebar.selectbox(label, data.seasons(), key=key)


def race_picker(data: F1Data, year: int, key: str) -> pd.Series:
    races = data.completed_races().query("year == @year").sort_values("round", ascending=False)
    labels = {r.raceId: f"R{r.round} · {r.name}" for r in races.itertuples()}
    race_id = st.sidebar.selectbox("Grand Prix", list(labels), format_func=labels.get, key=key)
    return races.set_index("raceId").loc[race_id].rename(race_id)
