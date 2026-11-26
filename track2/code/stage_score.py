"""Score sealed Track 2 predictions. Selection files are not modified."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA, MEMBERS, RESULTS, history_scales, load_targets, macro_mase, write_json

BOOT_SEED = 20261003
BOOT_DRAWS = 2000
BLOCK = 3


def _prepare(predictions: pd.DataFrame, split_name: str, history: str, destination: str) -> pd.DataFrame:
    targets = load_targets(split_name, history, destination)
    scored = predictions.merge(
        targets[["task_id", "target_date", "horizon_day", "actual"]],
        on=["task_id", "target_date", "horizon_day"],
        how="left",
        validate="many_to_one",
    )
    if scored["actual"].isna().any():
        raise RuntimeError(f"unmatched targets in {split_name} {history} {destination}")
    scales = history_scales(split_name, history, destination)
    scored = scored.merge(scales, on="task_id", how="left", validate="many_to_one")
    scored["scale_gap"] = (scored["mase7_scale"] - scored["s7_recomputed"]).abs()
    scored["mase7_scale"] = scored["s7_recomputed"]
    return scored


def _task_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for identity, part in scored.groupby(["market", "fold_id", "task_id"], sort=False):
        part = part.sort_values("horizon_day")
        actual = part["actual"].to_numpy(float)
        pred = part["prediction"].to_numpy(float)
        error = pred - actual
        absolute = np.abs(error)
        scale = float(part["mase7_scale"].iloc[0])
        mase = float(absolute.mean() / scale) if scale > 0 else np.nan
        high = np.zeros(len(part), dtype=bool)
        if "high_demand_threshold" in part.columns and part["high_demand_threshold"].notna().all():
            high = actual > float(part["high_demand_threshold"].iloc[0])
        row = {
            "market": identity[0],
            "fold_id": identity[1],
            "task_id": identity[2],
            "mase": mase,
            "mae": float(absolute.mean()),
            "rmsse": float(np.sqrt(np.mean(error**2)) / scale) if scale > 0 else np.nan,
            "wape": float(absolute.sum() / actual.sum()) if actual.sum() else np.nan,
            "bias_pct": float((pred.sum() - actual.sum()) / actual.sum()) if actual.sum() else np.nan,
            "high_mase": float(absolute[high].mean() / scale) if high.any() and scale > 0 else np.nan,
            "high_under": float(np.maximum(0.0, -error[high]).mean()) if high.any() else np.nan,
        }
        if {"q0.1", "q0.9"}.issubset(part.columns) and part["q0.1"].notna().all():
            low = part["q0.1"].to_numpy(float)
            high_q = part["q0.9"].to_numpy(float)
            row["covered"] = float(np.mean((actual >= low) & (actual <= high_q)))
            row["width"] = float(np.mean(high_q - low))
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize(tasks: pd.DataFrame, m1: pd.DataFrame | None, members: dict[str, pd.DataFrame] | None) -> dict:
    summary = {
        "tasks": int(len(tasks)),
        "markets": int(tasks["market"].nunique()),
        "mase": macro_mase(tasks),
        "mae": float(tasks.groupby("market")["mae"].mean().mean()),
        "rmsse": float(tasks.groupby("market")["rmsse"].mean().mean()),
        "wape": float(tasks.groupby("market")["wape"].mean().mean()),
        "bias_pct": float(tasks.groupby("market")["bias_pct"].mean().mean()),
        "high_mase": float(tasks.groupby("market")["high_mase"].mean().mean()),
        "origin_mase_std": float(tasks.groupby("fold_id").apply(macro_mase, include_groups=False).std(ddof=1)),
        "worst10_mase": float(tasks.loc[tasks["mase"] >= tasks["mase"].quantile(0.9), "mase"].mean()),
        "max_mase": float(tasks["mase"].max()),
    }
    if "covered" in tasks.columns:
        summary["coverage80"] = float(tasks["covered"].mean())
        summary["mean_width"] = float(tasks["width"].mean())
    if m1 is not None:
        paired = tasks.merge(m1[["task_id", "mase"]], on="task_id", suffixes=("", "_m1"))
        summary["win_rate_vs_m1"] = float((paired["mase"] < paired["mase_m1"]).mean())
    if members is not None:
        wide = tasks[["task_id", "mase"]].rename(columns={"mase": "ensemble"})
        for name, table in members.items():
            wide = wide.merge(table[["task_id", "mase"]].rename(columns={"mase": name}), on="task_id")
        best = wide[list(members)].min(axis=1)
        gap = wide[MEMBERS[0]] - best
        captured = (wide[MEMBERS[0]] - wide["ensemble"]) / gap.replace(0, np.nan)
        summary["oracle_gap_captured"] = float(captured.mean())
        corr = wide[list(members)].corr()
        summary["member_mase_corr"] = corr.round(4).to_dict()
    return summary


def _bootstrap(tables: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    folds = sorted(tables[names[0]]["fold_id"].unique())
    markets = sorted(tables[names[0]]["market"].unique())
    fold_index = {fold: index for index, fold in enumerate(folds)}
    market_index = {market: index for index, market in enumerate(markets)}
    cube = np.zeros((len(names), len(markets), len(folds)))
    for name_index, name in enumerate(names):
        for market, fold, mase in tables[name][["market", "fold_id", "mase"]].itertuples(index=False):
            cube[name_index, market_index[market], fold_index[fold]] = mase
    starts = np.arange(0, len(folds) - BLOCK + 1)
    needed = int(np.ceil(len(folds) / BLOCK))
    rng = np.random.default_rng(BOOT_SEED)
    chosen = rng.choice(len(starts), size=(BOOT_DRAWS, needed), replace=True)
    draws = {name: np.empty(BOOT_DRAWS) for name in names}
    for draw in range(BOOT_DRAWS):
        selected: list[int] = []
        for start in starts[chosen[draw]]:
            selected.extend(range(int(start), int(start) + BLOCK))
        selected = selected[:len(folds)]
        means = cube[:, :, selected].mean(axis=2).mean(axis=1)
        for name_index, name in enumerate(names):
            draws[name][draw] = means[name_index]
    rows = []
    for left, right in zip(names, names[1:]):
        diff = draws[right] - draws[left]
        rows.append({
            "contrast": f"{right}_minus_{left}",
            "mean_diff": float(diff.mean()),
            "ci95_low": float(np.quantile(diff, 0.025)),
            "ci95_high": float(np.quantile(diff, 0.975)),
        })
    return pd.DataFrame(rows)


def _score_tree(root, split_name: str, history: str, destination: str, reference: dict | None = None) -> list[dict]:
    rows = []
    task_tables = {}
    for path in sorted(root.glob("*/predictions.parquet")):
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions["destination"] == destination]
        scored = _prepare(predictions, split_name, history, destination)
        tasks = _task_table(scored)
        task_tables[path.parent.name] = tasks
        summary = _summarize(tasks, None, None)
        summary.update({
            "experiment": path.parent.name,
            "split": split_name,
            "history": history,
            "destination": destination,
            "max_scale_gap": float(scored["scale_gap"].max()),
            "exploratory": split_name == "outer_backtest",
        })
        rows.append(summary)
        tasks.to_parquet(path.parent / "task_mase.parquet", index=False)
    if reference and destination == "Macau" and history == "all" and split_name == "outer_backtest":
        m1 = reference["T2-01"]
        members = reference["members"]
        for row in rows:
            extra = _summarize(task_tables[row["experiment"]], m1, members if row["experiment"] != "T2-C1" else None)
            row.update({key: value for key, value in extra.items() if key not in row or key in {"win_rate_vs_m1", "oracle_gap_captured", "member_mase_corr"}})
    return rows


def _member_tasks() -> dict[str, pd.DataFrame]:
    from common import TRACK1, load_predictions
    frame = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS, destination="Macau")
    tables = {}
    for model in MEMBERS:
        part = frame[frame["model"] == model]
        scored = _prepare(part, "outer_backtest", "all", "Macau")
        tables[model] = _task_table(scored)
    return tables


def main() -> None:
    output = RESULTS / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    members = _member_tasks()
    m1 = pd.read_parquet(RESULTS / "outer_macau" / "T2-01" / "predictions.parquet")
    m1_tasks = _task_table(_prepare(m1, "outer_backtest", "all", "Macau"))
    reference = {"T2-01": m1_tasks, "members": members}
    rows = []
    rows.extend(_score_tree(RESULTS / "outer_macau", "outer_backtest", "all", "Macau", reference))
    rows.extend(_score_tree(RESULTS / "extension_hong_kong", "outer_backtest", "all", "HongKong"))
    rows.extend(_score_tree(RESULTS / "sensitivity_180", "outer_backtest", "180", "Macau"))
    rows.extend(_score_tree(RESULTS / "sensitivity_365", "outer_backtest", "365", "Macau"))
    rows.extend(_score_tree(RESULTS / "pandemic_stress", "pandemic_stress", "all", "Macau"))
    table = pd.DataFrame(rows)
    table.to_csv(output / "score_summary.csv", index=False)
    outer_tasks = {
        name: pd.read_parquet(RESULTS / "outer_macau" / name / "task_mase.parquet")
        for name in ("T2-01", "T2-03", "T2-04", "T2-05")
    }
    bootstrap = _bootstrap(outer_tasks, ["T2-01", "T2-03", "T2-04", "T2-05"])
    bootstrap.to_csv(output / "bootstrap_contrasts.csv", index=False)
    write_json(output / "score_manifest.json", {
        "exploratory_outer": True,
        "bootstrap_draws": BOOT_DRAWS,
        "bootstrap_seed": BOOT_SEED,
        "actuals_used_for_selection": False,
    })
    show = table[table["destination"].eq("Macau") & table["history"].eq("all") & table["split"].eq("outer_backtest")]
    print(show[["experiment", "mase", "win_rate_vs_m1", "worst10_mase"]].to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
