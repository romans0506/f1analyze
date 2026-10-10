"""Data layer: the Ergast-format CSV dump in ``data/`` plus a top-up from the
Jolpica API (the Ergast successor) for any race the CSVs don't cover yet.

Everything is keyed on string refs (``driverRef``, ``constructorRef``,
``circuitRef``) rather than the dump's numeric ids, because those are what the
API returns.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "jolpica"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
NA = r"\N"

FINISHED_PREFIXES = ("Finished", "+", "Lapped")


def is_classified_finish(status: str) -> bool:
    return isinstance(status, str) and status.startswith(FINISHED_PREFIXES)


@dataclass
class F1Data:
    races: pd.DataFrame       # one row per scheduled race (past and future)
    results: pd.DataFrame     # one row per driver per completed race
    qualifying: pd.DataFrame  # one row per driver per qualifying session
    sprints: pd.DataFrame     # one row per driver per sprint

    def completed_races(self) -> pd.DataFrame:
        done = self.races[self.races.raceId.isin(self.results.raceId)]
        return done.sort_values("date")

    def upcoming_races(self) -> pd.DataFrame:
        done = set(self.results.raceId)
        today = pd.Timestamp(date.today())
        future = self.races[~self.races.raceId.isin(done) & (self.races.date >= today - pd.Timedelta(days=1))]
        return future.sort_values("date")

    def seasons(self) -> list[int]:
        return sorted(self.completed_races().year.unique(), reverse=True)


# --------------------------------------------------------------------------- CSV

def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", na_values=NA, keep_default_na=True)


def _load_csv_dump() -> F1Data:
    drivers = _csv("drivers")
    drivers["driver"] = drivers.forename + " " + drivers.surname
    drivers = drivers[["driverId", "driverRef", "code", "driver", "nationality"]]
    constructors = _csv("constructors").rename(columns={"name": "team"})[["constructorId", "constructorRef", "team"]]
    circuits = _csv("circuits").rename(columns={"name": "circuit"})[
        ["circuitId", "circuitRef", "circuit", "location", "country", "lat", "lng"]]
    status = _csv("status")

    races = _csv("races").merge(circuits, on="circuitId", how="left")
    races["date"] = pd.to_datetime(races.date)
    races["start_utc"] = pd.to_datetime(races.date.dt.strftime("%Y-%m-%d") + " " + races.time.fillna("12:00:00"), utc=True)
    races["quali_date"] = pd.to_datetime(races.quali_date, errors="coerce")
    races["has_sprint"] = races.sprint_date.notna()
    races = races[["raceId", "year", "round", "name", "date", "start_utc", "quali_date", "has_sprint",
                   "circuitRef", "circuit", "location", "country", "lat", "lng"]]

    def expand(df: pd.DataFrame) -> pd.DataFrame:
        return (df.merge(drivers, on="driverId", how="left")
                  .merge(constructors, on="constructorId", how="left")
                  .drop(columns=["driverId", "constructorId"]))

    results = expand(_csv("results")).merge(status, on="statusId", how="left")
    results = results[["raceId", "driverRef", "code", "driver", "nationality", "constructorRef", "team",
                       "number", "grid", "position", "positionText", "positionOrder", "points", "laps", "status",
                       "fastestLapTime", "milliseconds"]]

    sprints = expand(_csv("sprint_results")).merge(status, on="statusId", how="left")
    sprints = sprints[["raceId", "driverRef", "code", "driver", "constructorRef", "team",
                       "grid", "position", "positionOrder", "points", "laps", "status"]]

    quali = expand(_csv("qualifying"))
    quali = quali[["raceId", "driverRef", "code", "driver", "constructorRef", "team", "position", "q1", "q2", "q3"]]
    return F1Data(races, results, quali, sprints)


# ----------------------------------------------------------------------- Jolpica

def _get(path: str, use_cache: bool) -> dict:
    """GET a Jolpica endpoint. Cached on disk when ``use_cache`` (only for
    finished events, whose data no longer changes)."""
    cache_file = CACHE_DIR / (path.strip("/").replace("/", "_") + ".json")
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    for attempt in range(4):
        resp = requests.get(f"{JOLPICA}/{path}", params={"limit": 100}, timeout=20)
        if resp.status_code == 429:  # rate limited (4 req/s, 500 req/h)
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()["MRData"]
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    raise RuntimeError(f"Jolpica kept rate-limiting {path}")


def _driver_cols(d: dict) -> dict:
    return {"driverRef": d["driverId"], "code": d.get("code"),
            "driver": f"{d['givenName']} {d['familyName']}", "nationality": d.get("nationality"),
            "number": pd.to_numeric(d.get("permanentNumber"), errors="coerce")}


def _team_cols(c: dict) -> dict:
    return {"constructorRef": c["constructorId"], "team": c["name"]}


def _result_rows(race_id: int, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        pos_text = r.get("positionText", "")
        out.append({
            "raceId": race_id, **_driver_cols(r["Driver"]), **_team_cols(r["Constructor"]),
            "grid": pd.to_numeric(r.get("grid"), errors="coerce"),
            "position": pd.to_numeric(pos_text, errors="coerce"),
            "positionText": pos_text,
            "positionOrder": int(r["position"]),
            "points": float(r.get("points", 0)),
            "laps": pd.to_numeric(r.get("laps"), errors="coerce"),
            "status": r.get("status"),
            "fastestLapTime": r.get("FastestLap", {}).get("Time", {}).get("time"),
            "milliseconds": pd.to_numeric(r.get("Time", {}).get("millis"), errors="coerce"),
        })
    return out


def _schedule_rows(year: int, races: pd.DataFrame) -> pd.DataFrame:
    """The season schedule from Jolpica, mapped onto existing raceIds."""
    known = races[races.year == year].set_index("round").raceId.to_dict()
    next_id = int(races.raceId.max()) + 1
    rows = []
    for r in _get(f"{year}/races/", use_cache=False)["RaceTable"]["Races"]:
        rnd = int(r["round"])
        race_id = known.get(rnd)
        if race_id is None:
            race_id, next_id = next_id, next_id + 1
        c = r["Circuit"]
        loc = c["Location"]
        rows.append({
            "raceId": race_id, "year": year, "round": rnd, "name": r["raceName"],
            "date": pd.Timestamp(r["date"]),
            "start_utc": pd.Timestamp(f"{r['date']} {r.get('time', '12:00:00Z')}").tz_convert("UTC")
                         if r.get("time") else pd.Timestamp(f"{r['date']} 12:00:00", tz="UTC"),
            "quali_date": pd.Timestamp(r["Qualifying"]["date"]) if "Qualifying" in r else pd.NaT,
            "has_sprint": "Sprint" in r,
            "circuitRef": c["circuitId"], "circuit": c["circuitName"], "location": loc.get("locality"),
            "country": loc.get("country"), "lat": float(loc["lat"]), "lng": float(loc["long"]),
        })
    return pd.DataFrame(rows)


def _top_up_from_api(data: F1Data, year: int) -> F1Data:
    """Fetch results/qualifying/sprints for races of ``year`` that the CSV
    dump is missing, and refresh that season's schedule."""
    schedule = _schedule_rows(year, data.races)
    races = pd.concat([data.races[data.races.year != year], schedule], ignore_index=True)

    have_results = set(data.results.raceId)
    have_quali = set(data.qualifying.raceId)
    have_sprint = set(data.sprints.raceId)
    today = pd.Timestamp(date.today())
    new_res, new_q, new_s = [], [], []

    for race in schedule.itertuples():
        base = f"{year}/{race.round}"
        race_done = race.date < today
        if race.raceId not in have_results and race.date <= today:
            got = _get(f"{base}/results/", use_cache=race_done)["RaceTable"]["Races"]
            if got:
                new_res += _result_rows(race.raceId, got[0]["Results"])
        if race.raceId not in have_quali and pd.notna(race.quali_date) and race.quali_date <= today:
            got = _get(f"{base}/qualifying/", use_cache=race_done)["RaceTable"]["Races"]
            for q in (got[0]["QualifyingResults"] if got else []):
                new_q.append({"raceId": race.raceId, **_driver_cols(q["Driver"]), **_team_cols(q["Constructor"]),
                              "position": int(q["position"]),
                              "q1": q.get("Q1"), "q2": q.get("Q2"), "q3": q.get("Q3")})
        if race.has_sprint and race.raceId not in have_sprint and race.date <= today:
            got = _get(f"{base}/sprint/", use_cache=race_done)["RaceTable"]["Races"]
            if got:
                new_s += _result_rows(race.raceId, got[0]["SprintResults"])

    def add(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return df
        extra = pd.DataFrame(rows)
        return pd.concat([df, extra[[c for c in df.columns if c in extra.columns]]], ignore_index=True)

    return F1Data(races, add(data.results, new_res), add(data.qualifying, new_q), add(data.sprints, new_s))


def load_lap_positions(race_id: int) -> pd.DataFrame:
    """Lap-by-lap positions from the CSV dump (1996 onwards), for races
    older than FastF1's coverage."""
    laps = pd.read_csv(DATA_DIR / "lap_times.csv", na_values=NA, usecols=["raceId", "driverId", "lap", "position", "milliseconds"])
    laps = laps[laps.raceId == race_id]
    drivers = _csv("drivers")[["driverId", "driverRef", "code", "surname"]]
    laps = laps.merge(drivers, on="driverId")
    laps["code"] = laps.code.fillna(laps.surname)
    return laps.drop(columns=["driverId", "surname"])


def load_data(refresh_from_api: bool = True) -> F1Data:
    """Load the full dataset. With ``refresh_from_api`` the latest season is
    topped up from Jolpica; network failures fall back to the CSVs alone."""
    data = _load_csv_dump()
    if refresh_from_api:
        try:
            data = _top_up_from_api(data, max(date.today().year, int(data.races.year.max())))
        except (requests.RequestException, RuntimeError, KeyError) as exc:
            print(f"[f1app] Jolpica top-up skipped: {exc}")

    # Season-wide context every view needs.
    # Pre-2005 drivers have no three-letter code; their surname labels them instead.
    for df in (data.results, data.qualifying, data.sprints):
        df["code"] = df.code.fillna(df.driver.str.split().str[-1])

    context = data.races[["raceId", "year", "round", "name", "date", "circuitRef"]]
    data.results = data.results.merge(context, on="raceId", how="left")
    data.results["finished"] = data.results.status.map(is_classified_finish)
    data.qualifying = data.qualifying.merge(context, on="raceId", how="left")
    data.sprints = data.sprints.merge(context, on="raceId", how="left")
    return data
