"""Lap-by-lap session data from FastF1 (live-timing archive, 2018 onwards)."""
from __future__ import annotations

import fastf1
import pandas as pd

from .data import ROOT

FASTF1_FROM = 2018
CACHE = ROOT / ".fastf1_cache"
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))
fastf1.set_log_level("ERROR")


def load_session(year: int, rnd: int, kind: str = "R", telemetry: bool = False):
    """``kind``: R (race), Q (qualifying), S (sprint), FP1..FP3."""
    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=True, telemetry=telemetry, weather=False, messages=False)
    return session


def lap_table(session) -> pd.DataFrame:
    laps = pd.DataFrame(session.laps).copy()
    laps["LapSeconds"] = laps.LapTime.dt.total_seconds()
    return laps[["Driver", "Team", "LapNumber", "LapSeconds", "Position", "Compound", "TyreLife",
                 "Stint", "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate"]].copy()


def race_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Representative racing laps: accurate, green flag, no in/out laps, and
    within 107% of the driver's median (drops safety-car outliers)."""
    clean = laps[laps.IsAccurate & laps.PitInTime.isna() & laps.PitOutTime.isna()
                 & (laps.TrackStatus.astype(str) == "1") & laps.LapSeconds.notna()]
    med = clean.groupby("Driver").LapSeconds.transform("median")
    return clean[clean.LapSeconds < med * 1.07]


def stints(laps: pd.DataFrame) -> pd.DataFrame:
    return (laps.dropna(subset=["Stint"])
                .groupby(["Driver", "Stint", "Compound"], as_index=False)
                .agg(start=("LapNumber", "min"), end=("LapNumber", "max")))


def fastest_lap_telemetry(session, driver: str) -> pd.DataFrame:
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap.LapTime):
        return pd.DataFrame()
    tel = lap.get_car_data().add_distance()
    tel["Driver"] = driver
    tel["LapTime"] = lap.LapTime.total_seconds()
    return tel[["Driver", "Distance", "Speed", "Throttle", "Brake", "nGear", "RPM", "LapTime"]]
