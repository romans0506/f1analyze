"""Turn a FastF1 session into a compact, frame-by-frame race replay: car
positions on the track map, running order, gaps, tyres, pit stops and flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRAME_SECONDS = 1.0
RETIRED = 10_000
FINISHED = 1_000  # progress value for cars that have taken the flag, minus classified position
OFF_TRACK = 250  # position units are 1/10 m: further than 25 m from the line = pit lane / run-off
MAX_SPEED = 1000  # 100 m/s in position units: no car covers more track than this per second
TRACK_STATUS = {"1": "green", "2": "yellow", "4": "sc", "5": "red", "6": "vsc", "7": "vsc_ending"}


def _secs(td) -> np.ndarray:
    return pd.to_timedelta(td).dt.total_seconds().to_numpy() if isinstance(td, pd.Series) else td.total_seconds()


def _rotation(session, xy: np.ndarray) -> tuple[float, pd.DataFrame | None]:
    """Map rotation in degrees and corner markers. Uses FastF1's circuit info
    when it exists, otherwise lines the track's long axis up horizontally."""
    try:
        info = session.get_circuit_info()
        return float(info.rotation), info.corners[["X", "Y", "Number", "Letter"]]
    except Exception:
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return -float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0]))), None


def _rotate(xy: np.ndarray, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return xy @ rot


def _rate_limit(prog: np.ndarray, max_step: float) -> np.ndarray:
    """Running maximum of ``prog`` that may rise by at most ``max_step`` per
    frame (catching up over the next frames instead of jumping)."""
    out = np.empty_like(prog)
    cur = prog[0]
    for k, v in enumerate(prog):
        cur = max(cur, min(v, cur + max_step))
        out[k] = cur
    return out


def reference_lap(session) -> pd.DataFrame:
    """X/Y/distance along the fastest clean lap: the track outline and the
    yardstick for how far round the lap each car is."""
    lap = session.laps.pick_quicklaps().pick_fastest()
    if lap is None or (isinstance(lap, pd.Series) and lap.empty):
        lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()[["X", "Y"]].dropna().reset_index(drop=True)
    pos = _start_at_timing_line(session, pos)
    step = np.hypot(pos.X.diff(), pos.Y.diff()).fillna(0)
    return pos.assign(Distance=step.cumsum())


def _start_at_timing_line(session, pos: pd.DataFrame) -> pd.DataFrame:
    """Re-order the reference lap so it begins where laps are timed. The line
    is where cars are at the moment each lap time is recorded."""
    marks = []
    for num, d in session.pos_data.items():
        crossings = _secs(session.laps[session.laps.DriverNumber == num].Time.dropna())
        if len(crossings) and len(d):
            ts = _secs(d.SessionTime)
            marks.append(np.column_stack([np.interp(crossings, ts, d.X), np.interp(crossings, ts, d.Y)]))
    if not marks:
        return pos
    line = np.median(np.vstack(marks), axis=0)
    k = int(((pos[["X", "Y"]].to_numpy(float) - line) ** 2).sum(axis=1).argmin())
    return pd.concat([pos.iloc[k:], pos.iloc[:k]], ignore_index=True)


def track_map(session) -> dict:
    """Just the outline + corners, for static track maps."""
    ref = reference_lap(session)
    xy = ref[["X", "Y"]].to_numpy(float)
    rot, corners = _rotation(session, xy)
    out = {"track": _rotate(xy, rot).round(0).astype(int).tolist(), "corners": []}
    if corners is not None:
        cxy = _rotate(corners[["X", "Y"]].to_numpy(float), rot)
        out["corners"] = [{"n": f"{int(n)}{l or ''}", "x": float(x), "y": float(y)}
                          for (x, y), n, l in zip(cxy, corners.Number, corners.Letter)]
    return out


def build_replay(session) -> dict:
    laps = session.laps
    results = session.results
    ref = reference_lap(session)
    ref_xy = ref[["X", "Y"]].to_numpy(float)
    ref_frac = (ref.Distance / ref.Distance.max()).to_numpy()

    start = float(_secs(laps[laps.LapNumber == 1].LapStartTime).min()) - 3
    end = float(np.nanmax(_secs(laps.Time))) + 20
    t = np.arange(start, end, FRAME_SECONDS)
    n = len(t)

    drivers, xs, ys, progress, first_lap, pit_start, grid_slot = [], [], [], [], [], [], []
    race_start = start + 3
    for num, pos in session.pos_data.items():
        row = results[results.DriverNumber == num]
        if row.empty or pos.empty:
            continue
        row = row.iloc[0]
        ts = _secs(pos.SessionTime)
        x = np.interp(t, ts, pos.X.to_numpy(float), left=np.nan, right=np.nan)
        y = np.interp(t, ts, pos.Y.to_numpy(float), left=np.nan, right=np.nan)

        dl = laps[laps.DriverNumber == num].sort_values("LapNumber")
        lap_end = _secs(dl.Time)
        lap_end = lap_end[~np.isnan(lap_end)]
        last_seen = lap_end.max() if len(lap_end) else start

        # Fraction round the lap = nearest point on the reference lap.
        pts = np.column_stack([x, y])
        ok = ~np.isnan(pts).any(axis=1)
        frac = np.full(n, np.nan)
        if ok.any():
            d2 = ((pts[ok, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2)
            near = d2.argmin(axis=1)
            on_line = d2.min(axis=1) < OFF_TRACK ** 2
            frac[np.flatnonzero(ok)[on_line]] = ref_frac[near[on_line]]
        done = np.searchsorted(np.sort(lap_end), t, side="right").astype(float)
        # Just after the line the nearest point may still be the end of the
        # reference lap (or vice versa); wrap those.
        since = t - np.concatenate([[start], np.sort(lap_end)])[done.astype(int)]
        frac = np.where((frac > 0.8) & (since < 15), frac - 1, frac)
        # Off the racing line (pit lane, pit-lane start): keep the last known
        # progress rather than trusting the nearest point.
        prog = pd.Series(done + frac).ffill().fillna(-0.1).to_numpy()
        # A nearest-point match can snap to a nearby part of the circuit; cap
        # how fast progress may grow so such jumps can't leapfrog the field.
        prog = _rate_limit(prog, MAX_SPEED * FRAME_SECONDS / ref.Distance.max())

        # Retired: stop the car once it's gone quiet after its last lap.
        status = str(row.Status)
        classified = status == "Finished" or status.startswith("+") or status == "Lapped"
        prog = np.maximum.accumulate(prog)
        out_from = finish_from = None
        official_gap = None
        if not classified:
            gone = t > last_seen + 30
            out_from = int(np.argmax(gone)) if gone.any() else None
            x[gone] = np.nan
            y[gone] = np.nan
            prog[gone] -= RETIRED  # sort behind every running car, by distance covered
        elif len(lap_end):
            # Once a car takes the flag, show the official classification
            # (which includes time penalties) and take the car off the map
            # after its slow-down.
            finished_at = lap_end.max()
            done_flag = t >= finished_at
            finish_from = int(np.argmax(done_flag)) if done_flag.any() else None
            if pd.notna(row.Position):
                prog[done_flag] = FINISHED - float(row.Position)
            gap_td = row.get("Time")
            if pd.notna(gap_td) and row.Position != 1:
                official_gap = round(gap_td.total_seconds(), 1)
            x[t > finished_at + 45] = np.nan
            y[t > finished_at + 45] = np.nan

        in_pit = np.zeros(n, dtype=bool)
        for p_in, p_out in zip(_secs(dl.PitInTime), _secs(dl.PitOutTime.shift(-1))):
            if not np.isnan(p_in):
                in_pit |= (t >= p_in) & (t <= (p_out if not np.isnan(p_out) else p_in + 30))

        compound = np.full(n, "", dtype=object)
        lap_start = _secs(dl.LapStartTime)
        for ls, c in zip(lap_start, dl.Compound):
            if not np.isnan(ls):
                compound[t >= ls] = c if isinstance(c, str) else ""

        drivers.append({
            "code": row.Abbreviation, "name": row.FullName, "team": row.TeamName,
            "color": "#" + (row.TeamColor if isinstance(row.TeamColor, str) and row.TeamColor else "888888"),
            "number": num, "finish": None if pd.isna(row.Position) else int(row.Position),
            "status": status, "classified": classified, "out_from": out_from,
            "finish_from": finish_from, "official_gap": official_gap,
            "pit": in_pit, "compound": compound,
        })
        xs.append(x)
        ys.append(y)
        progress.append(prog)
        first_lap.append(done == 0)
        lap1 = dl[dl.LapNumber == 1]
        started_in_pits = bool(len(lap1) and pd.notna(lap1.PitOutTime.iloc[0]))
        at_start = int(np.searchsorted(t, race_start))
        started_in_pits |= bool(np.isnan(frac[min(at_start, n - 1)]) and not np.isnan(x[min(at_start, n - 1)]))
        pit_start.append(started_in_pits)
        grid = row.GridPosition
        grid_slot.append(99 if started_in_pits or pd.isna(grid) or grid <= 0 else float(grid))

    X, Y, P = np.array(xs), np.array(ys), np.array(progress)

    # Pit-lane starters (flagged with a pit exit on lap 1, or off the racing
    # line when the lights go out) sit at pit exit, which can look like the
    # front of the field. Keep them behind every running car until they
    # complete lap 1.
    on_lap_one = np.array(first_lap)
    for i in np.flatnonzero(pit_start):
        running = np.delete(np.where(P > -1, P, np.inf), i, axis=0).min(axis=0)
        P[i] = np.where(on_lap_one[i] & np.isfinite(running), np.minimum(P[i], running - 0.001), P[i])

    # Before lights out, show the starting grid.
    pre_start = t < race_start
    P[:, pre_start] = -0.5 - np.array(grid_slot)[:, None] * 0.001

    # Rotate everything into the map frame.
    rot, corners = _rotation(session, ref_xy)
    flat = _rotate(np.column_stack([X.ravel(), Y.ravel()]), rot)
    X, Y = flat[:, 0].reshape(X.shape), flat[:, 1].reshape(Y.shape)
    track = _rotate(ref_xy, rot)

    # Running order and gap to the leader (time since the leader was at the
    # same point of the race).
    order = np.argsort(-P, axis=0, kind="stable")  # [rank, frame] -> driver index
    racing = (P >= 0) & (P < FINISHED - 100)
    lead_prog = np.maximum.accumulate(np.clip(np.where(racing, P, -1).max(axis=0), 0, None))
    lead_prog_u, idx = np.unique(lead_prog, return_index=True)
    lead_time = t[idx]
    gaps = np.where(racing, t[None, :] - np.interp(np.clip(P, 0, None), lead_prog_u, lead_time), np.nan)
    gaps = np.clip(gaps, 0, None)
    gaps[:, lead_prog < 0.05] = np.nan  # no meaningful gaps until the field gets moving

    leader_lap = np.clip(np.floor(lead_prog) + 1, 1, session.total_laps or np.inf)

    ts = session.track_status
    st_t = _secs(ts.Time)
    st_idx = np.searchsorted(st_t, t, side="right") - 1
    flags = [TRACK_STATUS.get(str(ts.Status.iloc[i]), "green") if i >= 0 else "green" for i in st_idx]

    def ints(a):
        return [None if np.isnan(v) else int(round(v)) for v in a]

    for i, d in enumerate(drivers):
        d["x"], d["y"] = ints(X[i]), ints(Y[i])
        d["gap"] = [None if np.isnan(g) else round(float(g), 1) for g in gaps[i]]
        d["pit"] = [int(b) for b in d["pit"]]
        # Compound only when it changes, to keep the payload small.
        comp = d.pop("compound")
        d["tyres"] = [[int(k), comp[k]] for k in range(n) if k == 0 or comp[k] != comp[k - 1]]

    corner_list = []
    if corners is not None:
        cxy = _rotate(corners[["X", "Y"]].to_numpy(float), rot)
        corner_list = [{"n": f"{int(num)}{let or ''}", "x": float(cx), "y": float(cy)}
                       for (cx, cy), num, let in zip(cxy, corners.Number, corners.Letter)]

    # Compress flags to change points too.
    flag_changes = [[k, f] for k, f in enumerate(flags) if k == 0 or f != flags[k - 1]]
    return {
        "event": session.event["EventName"], "year": int(session.event.year),
        "total_laps": int(session.total_laps or np.nanmax(laps.LapNumber)),
        "frame_seconds": FRAME_SECONDS, "frames": n,
        "track": track.round(0).astype(int).tolist(), "corners": corner_list,
        "drivers": drivers, "order": order.T.tolist(), "lap": leader_lap.astype(int).tolist(),
        "flags": flag_changes,
    }
