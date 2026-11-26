"""Pareto chart of Macau outer MASE(7) against cost per 1,000 tasks."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "track2" / "results" / "evaluation"
MAIN = ROOT / "track1" / "artifacts" / "scores" / "outer_main_table" / "main_table.csv"
COST = ROOT / "track1" / "evaluation" / "volume_breakeven" / "cost_per_task.csv"
SCORES = ROOT / "track2" / "results" / "evaluation" / "score_summary.csv"
API_PER_1000 = 0.963159

LABELS = {
    "seasonal_naive": "Seasonal naive",
    "ets": "ETS",
    "elasticnet": "Elastic Net",
    "ridge": "Ridge",
    "sarima": "SARIMA",
    "svr": "SVR",
    "lasso": "Lasso",
    "linear_regression": "Linear regression",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "transformer": "Small Transformer",
    "parrot_random_forest_800": "Parrot RF",
    "granite_patchtst_fm": "Granite / M1",
    "chronos_2": "Chronos-2",
    "timesfm_3": "TimesFM",
    "moirai": "Moirai",
    "patchtst": "PatchTST",
    "lag_llama": "Lag-Llama",
    "informer": "Informer",
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _points() -> list[dict]:
    accuracy = {row["model"]: float(row["Macau_MASE7"]) for row in _read(MAIN)}
    costs = {row["model"]: float(row["usd_per_task"]) * 1000 for row in _read(COST)}
    points = []
    for model, mase in accuracy.items():
        if model not in costs:
            continue
        points.append({"id": model, "label": LABELS.get(model, model), "group": "Track 1", "mase": mase, "usd_per_1000": costs[model]})
    track2 = {row["experiment"]: float(row["mase"]) for row in _read(SCORES) if row["destination"] == "Macau" and row["history"] == "all" and row["split"] == "outer_backtest"}
    granite = costs["granite_patchtst_fm"]
    chronos = costs["chronos_2"]
    timesfm = costs["timesfm_3"]
    trio = granite + chronos + timesfm
    baseline = costs["ets"] + costs["ridge"] + costs["elasticnet"]
    extras = [
        ("T2-C1", "Baseline ensemble", baseline, track2["T2-C1"]),
        ("T2-02", "M1+M2 equal", granite + chronos, track2["T2-02"]),
        ("T2-03", "Three-model equal", trio, track2["T2-03"]),
        ("T2-C2", "M1 plus fallback", granite, track2["T2-C2"]),
        ("T2-A1", "Deterministic coordinator", trio, track2["T2-01"]),
        ("T2-A2", "DeepSeek coordinator", trio + API_PER_1000, track2["T2-01"]),
    ]
    for identifier, label, cost, mase in extras:
        points.append({"id": identifier, "label": label, "group": "Track 2", "mase": mase, "usd_per_1000": cost})
    return points


def _frontier(points: list[dict]) -> list[dict]:
    kept = []
    for point in sorted(points, key=lambda item: (item["usd_per_1000"], item["mase"])):
        dominated = any(
            other["usd_per_1000"] <= point["usd_per_1000"] and other["mase"] <= point["mase"] and (other["usd_per_1000"] < point["usd_per_1000"] or other["mase"] < point["mase"])
            for other in points
            if other is not point
        )
        point["on_frontier"] = not dominated
        if point["on_frontier"]:
            kept.append(point)
    return kept


def _svg(points: list[dict], frontier: list[dict]) -> str:
    width, height = 1100, 720
    left, right, top, bottom = 90, 40, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [math.log10(point["usd_per_1000"]) for point in points]
    ys = [point["mase"] for point in points]
    xmin, xmax = min(xs) - 0.3, max(xs) + 0.3
    ymin, ymax = min(ys) - 0.08, max(ys) + 0.12

    def px(cost: float) -> float:
        return left + (math.log10(cost) - xmin) / (xmax - xmin) * plot_w

    def py(mase: float) -> float:
        return top + (ymax - mase) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="32" text-anchor="middle" font-family="sans-serif" font-size="18">Macau outer backtest: cost per 1,000 tasks vs MASE(7)</text>',
    ]
    for tick in range(math.floor(xmin), math.ceil(xmax) + 1):
        x = left + (tick - xmin) / (xmax - xmin) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12" font-family="sans-serif">10^{tick}</text>')
    y_start = math.floor(ymin * 5) / 5
    y = y_start
    while y <= ymax:
        yy = py(y)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{y:.1f}</text>')
        y += 0.2
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="sans-serif">USD per 1,000 tasks (log scale)</text>')
    parts.append(f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 22 {top + plot_h / 2})">Outer MASE(7)</text>')
    line = " ".join(f"{px(point['usd_per_1000']):.1f},{py(point['mase']):.1f}" for point in frontier)
    parts.append(f'<polyline points="{line}" fill="none" stroke="#b45309" stroke-width="2"/>')
    for point in points:
        color = "#1d4ed8" if point["group"] == "Track 2" else "#64748b"
        radius = 5.5 if point["on_frontier"] else 3.5
        parts.append(f'<circle cx="{px(point["usd_per_1000"]):.1f}" cy="{py(point["mase"]):.1f}" r="{radius}" fill="{color}" fill-opacity="0.85"><title>{point["label"]} {point["mase"]:.3f} ${point["usd_per_1000"]:.4g}</title></circle>')
        if point["on_frontier"] or point["group"] == "Track 2":
            parts.append(f'<text x="{px(point["usd_per_1000"]) + 7:.1f}" y="{py(point["mase"]) - 6:.1f}" font-size="11" font-family="sans-serif">{point["label"]}</text>')
    parts.append('<rect x="760" y="60" width="14" height="14" fill="#64748b"/>')
    parts.append('<text x="780" y="72" font-size="12" font-family="sans-serif">Track 1</text>')
    parts.append('<rect x="860" y="60" width="14" height="14" fill="#1d4ed8"/>')
    parts.append('<text x="880" y="72" font-size="12" font-family="sans-serif">Track 2</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    points = _points()
    frontier = _frontier(points)
    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / "pareto_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "group", "mase", "usd_per_1000", "on_frontier"])
        writer.writeheader()
        writer.writerows(points)
    (EVAL / "pareto_frontier.svg").write_text(_svg(points, frontier), encoding="utf-8")
    print("frontier", ", ".join(point["label"] for point in frontier))
    print("wrote", EVAL / "pareto_frontier.svg")


if __name__ == "__main__":
    main()
