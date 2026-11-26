"""Call DeepSeek on the sealed fault schedule. Numbers still come from the frozen rule."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import MEMBERS, RESULTS, TRACK1, history_scales, load_predictions, load_targets, write_json
from stage_a2 import NUMBER_RUN, SYSTEM, _post, read_env
from stage_faults import _deliver, _inject, _level_ok, _patterns

FAULT_REASON = {
    "R3": "import_conflict",
    "R4": "checkpoint_mismatch",
    "S1": "timeout",
    "S4": "origin_mismatch",
    "S5": "cuda_oom",
    "S2": "non_finite",
    "S3": "wrong_length",
}


def _member_report(values, recent: float, fault: str | None) -> dict:
    if values is None:
        return {"ok": False, "reason": FAULT_REASON.get(fault or "", "member_failed")}
    if len(values) != 28 or not np.isfinite(np.asarray(values, float)).all():
        return {"ok": False, "reason": FAULT_REASON.get(fault or "", "structure")}
    if not _level_ok(values, recent):
        return {"ok": False, "reason": "level"}
    return {"ok": True}


def _context(frame, seasonal, recent_map, patterns, event: dict) -> dict:
    task_id = event["task_id"]
    parts = frame[frame["task_id"] == task_id]
    vectors = {
        model: parts[parts["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float)
        for model in MEMBERS
    }
    vectors, _hard = _inject(vectors, event["fault"], event["market"], patterns, event["member"])
    recent = float(recent_map[task_id])
    naive = seasonal[seasonal["task_id"] == task_id].sort_values("horizon_day")["prediction"].to_numpy(float)
    prediction, detected, ok = _deliver(vectors, recent, naive, "T2-A1")
    reports = {}
    for model in MEMBERS:
        fault = event["fault"] if model == event["member"] else None
        reports[model] = _member_report(vectors[model], recent, fault)
    failed = [model for model, report in reports.items() if not report["ok"]]
    if not failed:
        path = "frozen_ensemble"
    elif ok and detected:
        path = "fallback"
    else:
        path = "undelivered"
    return {
        "task_id": task_id,
        "reports": reports,
        "path": path,
        "detected": bool(detected),
        "delivered": bool(ok),
        "prediction": None if prediction is None else prediction,
    }


def _tool_result(name: str, context: dict) -> dict:
    if name == "audit_task":
        return {"ok": True, "task_id": context["task_id"]}
    if name == "run_member":
        return {"members": context["reports"]}
    if name == "check_output":
        return {"members": context["reports"], "detected": context["detected"]}
    if name in {"apply_frozen_ensemble", "apply_frozen_residual", "record_fallback"}:
        return {"ok": context["delivered"], "path": context["path"], "residual_lambda": 0}
    if name == "write_explanation":
        return {"ok": True}
    return {"ok": False, "error": "unknown_tool"}


def _call(event: dict, context: dict, env: dict, meta: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM + " 如果工具报告成员失败，仍然只能调用冻结回退，不能自己写预测。"},
        {"role": "user", "content": json.dumps({
            "task_id": event["task_id"],
            "destination": meta["destination"],
            "market": meta["market"],
            "origin_date": meta["origin_date"],
            "fold_id": event["fold_id"],
            "fault_schedule_rate": event["rate"],
            "frozen_config_sha256": meta["config_sha256"],
        }, ensure_ascii=False)},
    ]
    calls = []
    violations = 0
    applied = False
    for _ in range(4):
        payload, elapsed = _post(env, messages)
        choice = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        content = choice.get("content") or ""
        if NUMBER_RUN.search(content):
            violations += 1
        calls.append({
            "task_id": event["task_id"],
            "rate": event["rate"],
            "fault": event["fault"],
            "model": payload.get("model") or env["DEEPSEEK_MODEL"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_seconds": elapsed,
            "http_status": 200,
        })
        tool_calls = choice.get("tool_calls") or []
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        if not tool_calls:
            break
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            if name in {"apply_frozen_ensemble", "apply_frozen_residual"}:
                applied = True
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(_tool_result(name, context), ensure_ascii=False),
            })
    return {"applied": applied, "violations": violations, "calls": calls, "served_model": calls[-1]["model"] if calls else None}


def main() -> None:
    output = RESULTS / "fault_injection"
    schedule_path = output / "schedule.json"
    raw = schedule_path.read_text(encoding="utf-8")
    body = raw[:-1] if raw.endswith("\n") else raw
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    saved = (output / "schedule_sha256.txt").read_text().strip()
    if digest != saved:
        raise SystemExit("fault schedule hash does not match the sealed file")
    schedule = json.loads(body)
    env = read_env()
    if not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is empty")
    frame = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS, destination="Macau")
    seasonal = pd.concat(
        (pd.read_parquet(path) for path in (TRACK1 / "outer_server_all" / "predictions" / "seasonal_naive" / "SN01" / "42").glob("*.parquet")),
        ignore_index=True,
    )
    seasonal = seasonal[seasonal["destination"] == "Macau"]
    recent_map = history_scales("outer_backtest", "all", "Macau").set_index("task_id")["recent28_mean"].to_dict()
    patterns = _patterns()
    meta_frame = frame.groupby("task_id", sort=False).agg(destination=("destination", "first"), market=("market", "first"), origin_date=("origin_date", "first")).reset_index()
    meta_frame["origin_date"] = pd.to_datetime(meta_frame["origin_date"]).dt.strftime("%Y-%m-%d")
    meta_map = meta_frame.set_index("task_id").to_dict(orient="index")
    config_sha = json.loads((RESULTS.parents[0] / "configs" / "track2_freeze_manifest.json").read_text())["ensemble_config_sha256"]
    contexts = []
    for event in schedule:
        context = _context(frame, seasonal, recent_map, patterns, event)
        meta = dict(meta_map[event["task_id"]])
        meta["config_sha256"] = config_sha
        contexts.append((event, context, meta))
    log_path = output / "llm_calls_fault.jsonl"
    if log_path.exists():
        log_path.unlink()
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_call, event, context, env, meta): (event, context) for event, context, meta in contexts}
        for future in as_completed(futures):
            event, context = futures[future]
            outcome = future.result()
            results.append({
                "rate": event["rate"],
                "task_id": event["task_id"],
                "fault": event["fault"],
                "member": event["member"],
                "applied": outcome["applied"],
                "violations": outcome["violations"],
                "detected_by_rule": context["detected"],
                "delivered_by_rule": context["delivered"],
                "path": context["path"],
                "served_model": outcome["served_model"],
            })
            with log_path.open("a", encoding="utf-8") as stream:
                for call in outcome["calls"]:
                    stream.write(json.dumps(call, ensure_ascii=False) + "\n")
    detail = pd.DataFrame(results)
    detail.to_csv(output / "a2_fault_calls.csv", index=False)
    summary_rows = []
    calls = pd.read_json(log_path, lines=True) if log_path.stat().st_size else pd.DataFrame()
    for rate, part in detail.groupby("rate"):
        summary_rows.append({
            "rate": float(rate),
            "injected": int(len(part)),
            "tool_applied": int(part["applied"].sum()),
            "numeric_violations": int(part["violations"].sum()),
            "served_model": sorted(part["served_model"].dropna().unique().tolist()),
        })
    write_json(output / "a2_fault_summary.json", {
        "schedule_sha256": digest,
        "llm_invoked": True,
        "non_fault_tasks": "reused sealed predictions",
        "numeric_source": "same frozen rule as T2-A1",
        "by_rate": summary_rows,
        "prompt_tokens": int(calls["prompt_tokens"].sum()) if len(calls) else 0,
        "completion_tokens": int(calls["completion_tokens"].sum()) if len(calls) else 0,
    })
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
