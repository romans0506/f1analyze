"""Race outcome model.

Two parts:
1. A gradient-boosted regressor predicting finishing position *given the
   driver finishes*, trained on classified finishers since ``TRAIN_FROM``
   with recent seasons weighted more heavily.
2. A Monte Carlo simulation that adds race-day noise and retirements (from
   driver/team DNF rates) and turns the predictions into win / podium /
   points probabilities.

Run ``python -m f1app.model`` for a walk-forward backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import F1Data, load_data
from .features import POST_QUALI_FEATURES, PRE_QUALI_FEATURES, build_features, dnf_probability

TRAIN_FROM = 2014       # start of the hybrid era
RECENCY_DECAY = 0.7     # weight multiplier per season back in time
N_SIMULATIONS = 20_000


@dataclass
class RaceModel:
    regressor: HistGradientBoostingRegressor
    features: list[str]
    noise_a: float   # race-day noise sd = noise_a + noise_b * predicted position
    noise_b: float
    dnf_prior: float

    @property
    def uses_grid(self) -> bool:
        return "grid" in self.features


def _training_rows(feats: pd.DataFrame, before: pd.Timestamp | None) -> pd.DataFrame:
    rows = feats[(feats.year >= TRAIN_FROM) & feats.positionOrder.notna()]
    if before is not None:
        rows = rows[rows.date < before]
    return rows


def train(feats: pd.DataFrame, use_grid: bool, before: pd.Timestamp | None = None) -> RaceModel:
    cols = POST_QUALI_FEATURES if use_grid else PRE_QUALI_FEATURES
    rows = _training_rows(feats, before)
    finishers = rows[rows.finished.astype(bool)]
    weights = RECENCY_DECAY ** (finishers.year.max() - finishers.year)

    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    reg.fit(finishers[cols], finishers.positionOrder, sample_weight=weights)

    # Race-day noise grows down the order (the front is more predictable than
    # the midfield): fit |residual| against the prediction on the last two
    # seasons. Residuals are in-sample, so scale up to stay honest — 1.25 was
    # the best-calibrated factor in the 2025-26 walk-forward backtest.
    recent = finishers[finishers.year >= finishers.year.max() - 1]
    pred = reg.predict(recent[cols])
    slope, intercept = np.polyfit(pred, np.abs(recent.positionOrder - pred), 1)
    dnf_prior = float(rows[rows.year >= rows.year.max() - 1].dnf.mean())
    return RaceModel(reg, cols, 1.25 * intercept, 1.25 * slope, dnf_prior)


def simulate(model: RaceModel, field: pd.DataFrame, n: int = N_SIMULATIONS, seed: int = 0) -> pd.DataFrame:
    """Simulate a race for the rows in ``field`` (one per driver, with feature
    columns). Returns one row per driver with outcome probabilities."""
    rng = np.random.default_rng(seed)
    mu = model.regressor.predict(field[model.features])
    p_dnf = dnf_probability(field, model.dnf_prior).to_numpy()

    k = len(field)
    sd = np.clip(model.noise_a + model.noise_b * mu, 1.0, None)
    score = mu[None, :] + rng.normal(0, 1, size=(n, k)) * sd[None, :]
    retired = rng.random((n, k)) < p_dnf[None, :]
    score[retired] = 1_000 + rng.random(retired.sum())  # retirements go to the back
    finish = score.argsort(axis=1).argsort(axis=1) + 1   # 1 = winner

    out = field[["driverRef", "constructorRef"]].copy()
    out["predicted_score"] = mu
    out["p_dnf"] = p_dnf
    out["p_win"] = (finish == 1).mean(axis=0)
    out["p_podium"] = (finish <= 3).mean(axis=0)
    out["p_points"] = (finish <= 10).mean(axis=0)
    out["expected_pos"] = finish.mean(axis=0)
    out["pos_p10"] = np.percentile(finish, 10, axis=0)
    out["pos_p90"] = np.percentile(finish, 90, axis=0)
    # The same range counting only races the driver finishes.
    finished_only = np.where(retired, np.nan, finish)
    out["fin_p10"] = np.nanpercentile(finished_only, 10, axis=0)
    out["fin_p90"] = np.nanpercentile(finished_only, 90, axis=0)
    # Full finishing-position distribution, for the heatmap.
    dist = np.stack([(finish == p).mean(axis=0) for p in range(1, k + 1)], axis=1)
    out["position_dist"] = list(dist)
    # Predicted order follows the model's point prediction; the simulation
    # only supplies the probabilities around it.
    out = out.sort_values("predicted_score").reset_index(drop=True)
    out["predicted_pos"] = np.arange(1, k + 1)
    return out


# --------------------------------------------------------------- upcoming race

def entry_list(data: F1Data, race_id: int) -> tuple[pd.DataFrame, bool]:
    """Drivers for ``race_id`` and whether real grid positions are known.
    Uses the qualifying result if it exists, else the last race's line-up."""
    q = data.qualifying[data.qualifying.raceId == race_id]
    if len(q):
        e = q[["driverRef", "constructorRef", "position"]].rename(columns={"position": "grid"})
        return e.assign(raceId=race_id), True
    last = data.completed_races().raceId.iloc[-1]
    e = data.results[data.results.raceId == last][["driverRef", "constructorRef"]]
    return e.assign(raceId=race_id, grid=np.nan), False


def predict_race(data: F1Data, race_id: int, n: int = N_SIMULATIONS) -> tuple[pd.DataFrame, RaceModel]:
    entries, grid_known = entry_list(data, race_id)
    feats = build_features(data, entries)
    model = train(feats, use_grid=grid_known)
    field = feats[feats.raceId == race_id]
    sim = simulate(model, field, n=n)
    info = (data.results.drop_duplicates("driverRef", keep="last").set_index("driverRef")[["code", "driver"]])
    teams = data.results.drop_duplicates("constructorRef", keep="last").set_index("constructorRef").team
    sim = sim.join(info, on="driverRef")
    sim["team"] = sim.constructorRef.map(teams)
    sim = sim.merge(field[["driverRef"] + model.features], on="driverRef", how="left")
    return sim, model


# ------------------------------------------------------------------- backtest

def backtest(data: F1Data, seasons: list[int], use_grid: bool = True, n: int = 4_000) -> pd.DataFrame:
    """Walk-forward: for every race in ``seasons`` train on all earlier races
    only, predict, and score against what really happened."""
    feats = build_features(data)
    rows = []
    for race in data.completed_races().query("year in @seasons").itertuples():
        model = train(feats, use_grid=use_grid, before=race.date)
        field = feats[feats.raceId == race.raceId]
        sim = simulate(model, field, n=n)
        actual = field.set_index("driverRef").positionOrder
        sim["actual"] = sim.driverRef.map(actual)
        top3_pred = set(sim.head(3).driverRef)
        top3_real = set(sim.nsmallest(3, "actual").driverRef)
        grid_order = field.sort_values("grid").driverRef.tolist()
        rows.append({
            "raceId": race.raceId, "year": race.year, "round": race.round, "name": race.name,
            "winner_pred": sim.iloc[0].driverRef,
            "winner_real": sim.nsmallest(1, "actual").driverRef.iloc[0],
            "p_win_real_winner": float(sim.loc[sim.actual == 1, "p_win"].iloc[0]),
            "podium_hits": len(top3_pred & top3_real),
            "spearman": sim.predicted_pos.corr(sim.actual, method="spearman"),
            "grid_spearman": field.grid.corr(field.positionOrder, method="spearman"),
            "pole_won": grid_order[0] == sim.nsmallest(1, "actual").driverRef.iloc[0],
        })
    bt = pd.DataFrame(rows)
    bt["winner_hit"] = bt.winner_pred == bt.winner_real
    return bt


def summarize_backtest(bt: pd.DataFrame) -> dict:
    return {
        "races": len(bt),
        "winner_accuracy": bt.winner_hit.mean(),
        "pole_baseline_accuracy": bt.pole_won.mean(),
        "avg_podium_hits": bt.podium_hits.mean(),
        "avg_spearman": bt.spearman.mean(),
        "grid_baseline_spearman": bt.grid_spearman.mean(),
        "avg_p_win_actual_winner": bt.p_win_real_winner.mean(),
    }


if __name__ == "__main__":
    d = load_data()
    year = int(d.completed_races().year.max())
    for grid in (False, True):
        bt = backtest(d, [year - 1, year], use_grid=grid)
        label = "post-qualifying" if grid else "pre-qualifying"
        print(f"\n== {label} backtest, {year - 1}-{year}")
        for k, v in summarize_backtest(bt).items():
            print(f"  {k:28s} {v:.3f}" if isinstance(v, float) else f"  {k:28s} {v}")
    upcoming = d.upcoming_races()
    if len(upcoming):
        r = upcoming.iloc[0]
        sim, m = predict_race(d, int(r.raceId))
        print(f"\n== Prediction: {r['name']} {r.date.date()} (grid known: {m.uses_grid})")
        print(sim[["predicted_pos", "code", "team", "p_win", "p_podium", "p_points", "expected_pos"]]
              .head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
