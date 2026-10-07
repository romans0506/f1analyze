# F1 Analyze

Interactive Formula 1 analytics for every Grand Prix since 1950, plus a
supervised-learning model that predicts the next race with win / podium /
points probabilities.

## Pages

| Page | What it shows |
|---|---|
| **Next GP prediction** | Win, podium and points probability per driver, predicted finishing order with an 80% range, the full finishing-position distribution, the features behind each prediction, and the history of the circuit. Updates to use the real grid as soon as qualifying is done. |
| **Season overview** | Drivers' and constructors' standings, points progression, a results grid of every round, and team points per round, for any season. |
| **Grand Prix analysis** | Results and grid→finish changes for any race. From 2018 onwards it also shows lap-by-lap positions, race pace, tyre strategy, gaps to the leader, a telemetry comparison of two drivers' fastest laps, and qualifying gaps. |
| **Driver head-to-head** | Any two drivers, over a season or their whole careers: qualifying and race head-to-head, results over time, and who is stronger at which circuits. |
| **Model accuracy** | A walk-forward backtest that compares the model with the "grid order = finishing order" baseline. |

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
