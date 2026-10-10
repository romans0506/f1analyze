import json
from pathlib import Path

import streamlit as st

from f1app import sessions
from views.common import get_data, get_replay, is_simple

PLAYER = (Path(__file__).resolve().parent.parent / "f1app" / "replay_player.html").read_text(encoding="utf-8")

data = get_data()
st.title("📺 Race replay")
if is_simple():
    st.markdown("Watch any race again from above: every dot is a car in its team colour. Press **Play**, "
                "speed it up, or **click a driver** in the list to follow them.")
else:
    st.caption("Positions from the official timing feed (≈4 Hz, resampled to 1 s and interpolated). "
               "Running order is computed from distance round the lap, so it updates mid-lap; gaps are the "
               "time since the leader passed the same point.")

races = data.completed_races().query("year >= @sessions.FASTF1_FROM").sort_values("date", ascending=False)
years = sorted(races.year.unique(), reverse=True)
# Another page may have asked for a specific race ("Watch this race").
wanted = st.session_state.pop("replay_race", None)
if wanted in set(races.raceId):
    st.session_state["replay_year"] = int(races.set_index("raceId").loc[wanted, "year"])
    st.session_state["replay_race_pick"] = wanted
a, b = st.columns([1, 3])
year = a.selectbox("Season", years, key="replay_year")
in_year = races[races.year == year]
labels = {r.raceId: f"Round {r.round} · {r.name}" for r in in_year.itertuples()}
if st.session_state.get("replay_race_pick") not in labels:
    st.session_state.pop("replay_race_pick", None)
race_id = b.selectbox("Race", list(labels), format_func=labels.get, key="replay_race_pick")
race = in_year.set_index("raceId").loc[race_id]

try:
    rep = get_replay(int(year), int(race["round"]))
except Exception as exc:
    st.error(f"No position data for this race yet ({exc}). It usually appears about an hour after the chequered flag.")
    st.stop()

html = PLAYER.replace("__REPLAY_DATA__", json.dumps(rep, separators=(",", ":")))
st.iframe(html, height=780)

with st.expander("Keyboard shortcuts & legend"):
    st.markdown("""
- **Space**: play / pause · **← / →**: jump 10 seconds
- Coloured ring in the list = current tyre: **S** soft (red), **M** medium (yellow), **H** hard (white),
  **I** intermediate (green), **W** wet (blue)
- **PIT** = in the pit lane · **OUT** = retired from the race
- A banner at the top shows yellow flags, safety cars and red flags as they happened
""")
