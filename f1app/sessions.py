"""Lap-by-lap session data from FastF1 (live-timing archive, 2018 onwards)."""
from __future__ import annotations

import fastf1
import numpy as np
import pandas as pd

from .data import ROOT

FASTF1_FROM = 2018
FUEL_EFFECT = 0.035  # typical lap-time gain per lap from burning fuel (s)
DRY_AND_WET = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
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


def degradation(clean: pd.DataFrame, fuel_effect: float = FUEL_EFFECT, min_laps: int = 6) -> dict:
    """Tyre wear from clean race laps. Lap times are fuel-corrected (later laps
    are flattered by a lighter car), then a straight line is fitted against
    tyre age per compound and per stint."""
    laps = clean.dropna(subset=["TyreLife", "Compound", "Stint"])
    laps = laps[laps.Compound.isin(DRY_AND_WET)].copy()
    laps["Corrected"] = laps.LapSeconds + fuel_effect * (laps.LapNumber - 1)

    compound_fit = {}
    for comp, g in laps.groupby("Compound"):
        if len(g) >= 10 and g.TyreLife.nunique() > 3:
            slope, icept = np.polyfit(g.TyreLife, g.Corrected, 1)
            xs = np.array([g.TyreLife.min(), g.TyreLife.max()])
            compound_fit[comp] = {"slope": slope, "x": xs, "y": icept + slope * xs}

    rows = []
    for (drv, stint, comp), g in laps.groupby(["Driver", "Stint", "Compound"]):
        if len(g) >= min_laps:
            slope = np.polyfit(g.TyreLife, g.Corrected, 1)[0]
            rows.append({"Driver": drv, "Stint": int(stint), "Compound": comp, "laps": len(g),
                         "deg_per_lap": slope, "avg_pace": g.LapSeconds.mean(),
                         "label": f"{drv} · stint {int(stint)}"})
    return {"laps": laps, "compound_fit": compound_fit, "stints": pd.DataFrame(rows)}
