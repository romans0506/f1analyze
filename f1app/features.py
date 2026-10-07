"""Per-driver, per-race features built only from information available
*before* that race (every rolling stat is shifted by one race)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import F1Data

FORM_WINDOW = 5
DNF_WINDOW = 10

PRE_QUALI_FEATURES = [
    "drv_form", "drv_pts", "drv_quali", "drv_dnf", "team_form", "team_quali",
    "track_hist", "champ_rank", "sprint_pos",
]
POST_QUALI_FEATURES = PRE_QUALI_FEATURES + ["grid"]


def _rolling_prior(df: pd.DataFrame, by: str, col: str, window: int) -> pd.Series:
    """Mean of ``col`` over the previous ``window`` rows of each ``by`` group."""
    return df.groupby(by)[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def build_features(data: F1Data, entries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per (raceId, driverRef) with model features.

    ``entries`` optionally adds rows for a race that hasn't happened yet
    (columns: raceId, driverRef, constructorRef, grid); their features are
    computed from all races before it, exactly like historical rows.
    """
    res = data.results[["raceId", "date", "year", "circuitRef", "driverRef", "constructorRef",
                        "grid", "positionOrder", "points", "finished"]].copy()
    if entries is not None:
        race = data.races.set_index("raceId").loc[entries.raceId.iloc[0]]
        future = entries.assign(date=race.date, year=race.year, circuitRef=race.circuitRef,
                                positionOrder=np.nan, points=np.nan, finished=np.nan)
        res = pd.concat([res, future[res.columns]], ignore_index=True)

    quali = data.qualifying[["raceId", "driverRef", "position"]].rename(columns={"position": "quali_pos"})
    res = res.merge(quali, on=["raceId", "driverRef"], how="left")
    # Before ~1994 there's no qualifying table; the grid is the next best thing.
    res["quali_pos"] = res.quali_pos.fillna(res.grid.where(res.grid > 0))
    sprint = data.sprints[["raceId", "driverRef", "positionOrder"]].rename(columns={"positionOrder": "sprint_pos"})
    res = res.merge(sprint, on=["raceId", "driverRef"], how="left")

    res = res.sort_values(["date", "raceId"]).reset_index(drop=True)
    res["dnf"] = 1.0 - res.finished.astype(float)
    res.loc[res.positionOrder.isna(), "dnf"] = np.nan

    # Driver form.
    res["drv_form"] = _rolling_prior(res, "driverRef", "positionOrder", FORM_WINDOW)
    res["drv_pts"] = _rolling_prior(res, "driverRef", "points", FORM_WINDOW)
    res["drv_quali"] = _rolling_prior(res, "driverRef", "quali_pos", FORM_WINDOW)
    res["drv_dnf"] = _rolling_prior(res, "driverRef", "dnf", DNF_WINDOW)

    # Team form: average both cars per race, then roll over the team's races.
    team = (res.groupby(["constructorRef", "raceId", "date"], as_index=False)
               .agg(team_pos=("positionOrder", "mean"), team_q=("quali_pos", "mean"), team_dnf=("dnf", "mean"))
               .sort_values(["date", "raceId"]))
    team["team_form"] = _rolling_prior(team, "constructorRef", "team_pos", FORM_WINDOW)
    team["team_quali"] = _rolling_prior(team, "constructorRef", "team_q", FORM_WINDOW)
    team["team_dnf"] = _rolling_prior(team, "constructorRef", "team_dnf", DNF_WINDOW)
    res = res.merge(team[["constructorRef", "raceId", "team_form", "team_quali", "team_dnf"]],
                    on=["constructorRef", "raceId"], how="left")

    # Track history: driver's mean finish at this circuit in earlier years.
    res = res.sort_values(["date", "raceId"]).reset_index(drop=True)
    res["track_hist"] = (res.groupby(["driverRef", "circuitRef"]).positionOrder
                            .transform(lambda s: s.shift(1).expanding().mean()))

    # Championship rank going into the race (1 = leader).
    res["season_pts_before"] = (res.groupby(["year", "driverRef"]).points
                                   .transform(lambda s: s.shift(1).fillna(0).cumsum()))
    res["champ_rank"] = res.groupby("raceId").season_pts_before.rank(ascending=False, method="min")
    first_round = res.groupby("year").date.transform("min") == res.date
    res.loc[first_round, "champ_rank"] = np.nan

    # Rookies / new teams: fall back sensibly instead of leaving gaps.
    res["drv_quali"] = res.drv_quali.fillna(res.team_quali)
    res["drv_form"] = res.drv_form.fillna(res.team_form)
    res["drv_dnf"] = res.drv_dnf.fillna(res.team_dnf)
    res.loc[res.grid == 0, "grid"] = 22  # 0 = pit-lane start; NaN = grid not set yet

    return res


def dnf_probability(features: pd.DataFrame, prior: float, strength: float = 4.0) -> pd.Series:
    """Blend driver and team retirement rates, shrunk towards ``prior``."""
    raw = features[["drv_dnf", "team_dnf"]].mean(axis=1).fillna(prior)
    return (raw * DNF_WINDOW + prior * strength) / (DNF_WINDOW + strength)
