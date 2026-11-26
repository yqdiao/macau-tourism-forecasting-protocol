"""T2-X0: inventory member prediction files. Does not open targets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import MEMBERS, RESULTS, TRACK1, prediction_files, write_json

LAYOUTS = {
    "B": TRACK1 / "foundation_b_default",
    "O": TRACK1 / "foundation_outer_default",
}


def _origin_from_path(path: Path) -> str | None:
    for part in path.parts:
        if len(part) >= 11 and part[0] in "ABO" and part[3] == "_":
            return part.split("_", 1)[1]
    return None


def inventory_split(label: str, base: Path) -> pd.DataFrame:
    rows = []
    for model, path in prediction_files(base, MEMBERS):
        frame = pd.read_parquet(path)
        macau = frame[frame["destination"] == "Macau"]
        markets = macau["market"].nunique()
        expected = markets * 28
        horizons = sorted(macau["horizon_day"].unique().tolist()) if len(macau) else []
        finite = bool(np.isfinite(macau["prediction"].to_numpy(float)).all()) if len(macau) else False
        q_ok = True
        if {"q0.1", "q0.5", "q0.9"}.issubset(macau.columns) and len(macau):
            q_ok = bool(((macau["q0.1"] <= macau["q0.5"]) & (macau["q0.5"] <= macau["q0.9"])).all())
        origin = pd.to_datetime(macau["origin_date"]).dt.strftime("%Y-%m-%d").unique().tolist() if len(macau) else []
        folder_origin = _origin_from_path(path)
        rows.append(
            {
                "split": label,
                "model": model,
                "path": str(path.relative_to(TRACK1.parents[1])),
                "fold_id": macau["fold_id"].iloc[0] if len(macau) else None,
                "macau_rows": int(len(macau)),
                "macau_markets": int(markets),
                "rows_match_markets_x_28": int(len(macau) == expected and expected > 0),
                "horizons_1_to_28": int(horizons == list(range(1, 29))),
                "prediction_finite": int(finite),
                "quantiles_monotone": int(q_ok),
                "origin_matches_folder": int(origin == [folder_origin]),
                "folder_origin": folder_origin,
                "file_origins": ",".join(origin),
            }
        )
    return pd.DataFrame(rows)


def key_mismatches(base: Path) -> pd.DataFrame:
    frames = []
    for model in MEMBERS:
        files = [path for name, path in prediction_files(base, [model])]
        parts = [pd.read_parquet(path, columns=["destination", "market", "origin_date", "horizon_day", "fold_id", "task_id"]) for path in files]
        frame = pd.concat(parts, ignore_index=True)
        frame["model"] = model
        frames.append(frame)
    wide = frames[0]
    for other in frames[1:]:
        wide = wide.merge(
            other,
            on=["destination", "market", "origin_date", "horizon_day", "fold_id", "task_id"],
            how="outer",
            indicator=True,
        )
        bad = wide[wide["_merge"] != "both"]
        if len(bad):
            return bad
        wide = wide.drop(columns=["_merge", "model_x", "model_y"], errors="ignore")
    return pd.DataFrame(columns=["destination", "market", "origin_date", "horizon_day", "fold_id", "task_id"])


def main() -> None:
    output = RESULTS / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    tables = [inventory_split(label, path) for label, path in LAYOUTS.items()]
    inventory = pd.concat(tables, ignore_index=True)
    inventory.to_csv(output / "member_inventory.csv", index=False)
    mismatches = []
    for label, path in LAYOUTS.items():
        bad = key_mismatches(path)
        if len(bad):
            bad = bad.copy()
            bad["split"] = label
            mismatches.append(bad)
    mismatch_frame = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()
    mismatch_frame.to_csv(output / "key_mismatches.csv", index=False)
    b = inventory[inventory["split"] == "B"]
    o = inventory[inventory["split"] == "O"]
    summary = {
        "B_files_per_member": b.groupby("model").size().to_dict(),
        "O_files_per_member": o.groupby("model").size().to_dict(),
        "failed_rows": int((inventory[["rows_match_markets_x_28", "horizons_1_to_28", "prediction_finite", "origin_matches_folder"]].min(axis=1) == 0).sum()),
        "quantile_failures": int((inventory["quantiles_monotone"] == 0).sum()),
        "key_mismatches": int(len(mismatch_frame)),
    }
    write_json(output / "summary.json", summary)
    print(json_dumps(summary))
    if summary["failed_rows"] or summary["key_mismatches"]:
        raise SystemExit("T2-X0 failed")
    if list(summary["B_files_per_member"].values()) != [12, 12, 12] and set(summary["B_files_per_member"].values()) != {12}:
        raise SystemExit(f"B coverage is not 12/12: {summary['B_files_per_member']}")
    if set(summary["O_files_per_member"].values()) != {33}:
        raise SystemExit(f"O coverage is not 33/33: {summary['O_files_per_member']}")


def json_dumps(payload: dict) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
