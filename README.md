# F1 Analyze

Interactive Formula 1 analytics for every Grand Prix since 1950, plus a
supervised-learning model that predicts the next race with win / podium /
points probabilities.

## Two modes

Switch between them at the top of the sidebar. The choice is kept in the URL,
so `?mode=expert` or `?mode=simple` links open in that mode.

### 🟢 Simple: for everyone

| Page | What it shows |
|---|---|
| **Race weekend** | A live countdown to lights out, the predicted podium, "pick your driver" with their chances in plain words ("about 1 in 4"), the track map with corner numbers, and track facts. |
| **Race replay** | The race from above: every car moves round the track in team colours, with a live running order, gaps, tyres, pit stops, safety cars and flags. You can play, pause, change the speed, scrub, jump to a lap, and click a driver to follow them. |
| **Last race recap** | The race told as a short story: winner, biggest climber, retirements, and the championship swing. Also shows the podium, places gained or lost, and the full result. |
| **Championship** | The podium, who can still mathematically win the title, an animated race-by-race standings chart, and the team standings. |

### 🔬 Expert: every stat

| Page | What it shows |
|---|---|
| **Next GP prediction** | Win, podium and points probabilities, the predicted order with an 80% range, the finishing-position distribution, the features behind each prediction, and circuit history. |
| **Season overview** | Standings, points progression, a results grid of every round, and team points per round, for any season since 1950. |
| **Grand Prix analysis** | Results, lap-by-lap positions, race pace, tyre strategy, **tyre degradation** (fuel-corrected seconds lost per lap, per compound and per stint), gaps to the leader, fastest-lap telemetry comparison and qualifying. |
| **Race replay** | The same player as in Simple mode. |
| **Driver head-to-head** | Any two drivers over a season or their careers. |
| **Model accuracy** | A walk-forward backtest against the "grid order" baseline. |

### About the replay

The replay uses the official position feed (about 4 Hz), resampled to one
frame per second and smoothed in the browser. The running order comes from
each car's distance round the lap, so it updates mid-lap rather than only at
the timing line. It handles cars starting from the pit lane, and once each car
takes the flag it switches to the official classification, including
penalties. Data for a new race is available about an hour after the chequered
flag. True *live* positions during a session would need an F1 TV subscription
or a paid OpenF1 account, so they aren't included.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first time you open a race's lap or telemetry view, FastF1 downloads that
session (about 30–60 s). After that it loads from `.fastf1_cache/`.

## Data

- `data/*.csv` is the Ergast-format historical database (1950 → present).
- Any race the CSVs don't cover yet is fetched automatically from the
  [Jolpica API](https://github.com/jolpica/jolpica-f1) (the successor to
  Ergast) when the app starts, and cached in `data/cache/`.
- Lap times, tyres and telemetry come from [FastF1](https://docs.fastf1.dev/)
  (2018 onwards).

## The model (`f1app/model.py`, `f1app/features.py`)

1. **Features**: every feature is computed only from races *before* the one
   being predicted. They are the driver's average finish, points and qualifying
   over the last 5 races, their DNF rate over the last 10, the team's average
   finish and qualifying, the driver's past results at the circuit, their
   championship position, the sprint result (on sprint weekends) and, once
   qualifying is done, the starting grid.
2. **Regressor**: `HistGradientBoostingRegressor` predicts finishing position
   for classified finishers. It's trained on races since 2014, and each earlier
   season is weighted ×0.7.
3. **Monte Carlo**: 20,000 simulated races. Each one adds race-day noise to the
   predictions (more noise further down the order) and random retirements based
   on recent reliability. Counting the outcomes gives the probabilities.

Run the backtest from the command line:

```bash
python -m f1app.model
```

Walk-forward results for 2025–2026 (40 races, model retrained before each race):

| | Before qualifying | After qualifying | Baseline: grid order |
|---|---|---|---|
| Winner correct | 28% | 57% | 70% (pole-sitter won) |
| Podium finishers named (of 3) | 1.68 | 2.02 | — |
| Rank correlation with result | 0.57 | 0.65 | 0.65 |
| Avg win chance given to the actual winner | 22% | 38% | — |

The grid alone is a very strong predictor in F1. After qualifying, the model
puts the finishing order in about the same place as the grid, and its value is
in the calibrated probabilities. Before qualifying, the model is the only
forecast you have.
