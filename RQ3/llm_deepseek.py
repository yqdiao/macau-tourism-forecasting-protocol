"""DeepSeek OpenAI-compatible judge for RQ3 (reads RQ3/.env)."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import CACHE, COMBOS, LABEL, POOL, REPO

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_rq3_env(path: Path = ENV_PATH) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    # process env overrides
    for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_TEMPERATURE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if not env.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(f"DEEPSEEK_API_KEY missing in {path}")
    env.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    env.setdefault("DEEPSEEK_MODEL", "deepseek-chat")
    env.setdefault("DEEPSEEK_TEMPERATURE", "0")
    return env


SYSTEM = (
    "You are a constrained forecasting ensemble judge for Macau tourism origin-market forecasts. "
    "Use ONLY the provided historical summary. Do not invent models or use future outcomes. "
    "Reply with a single JSON object and no markdown."
)


def build_select_weight_prompt(feat: dict) -> str:
    return (
        "Task: choose exactly 3 models from the candidate pool and assign non-negative weights that sum to 1.\n"
        f"Candidate pool ids: {POOL}\n"
        f"Legal combo ids C1-C9 and members: {json.dumps({k: COMBOS[k] for k in [f'C{i}' for i in range(1,10)]})}\n"
        f"Market={feat['market']} split={feat['split_name']} origin={feat['origin_date']}\n"
        f"Holiday days in next 28: {feat['holiday_days']}\n"
        f"Recent demand (past 28d): mean={feat.get('demand_mean')} std={feat.get('demand_std')} "
        f"zero_share={feat.get('zero_share')} peak_ratio={feat.get('peak_ratio')}\n"
        f"Discounted single-model MAE on completed history H: {json.dumps(feat['single_mae'])}\n"
        f"Discounted combo MAE on H for C1-C9: {json.dumps(feat['combo_mae'])}\n"
        f"Forecast dispersion (Mode1): {feat.get('dispersion')}\n"
        f"Stat-Select suggestion: combo={feat['stat_combo']} members={feat['stat_members']} "
        f"stat_weights={feat['stat_weights']}\n"
        "Output JSON schema:\n"
        '{"selected_models":["id1","id2","id3"],"weights":[w1,w2,w3],'
        '"reason_codes":["..."],"evidence_ids":["..."],"combo_id":"C?"}\n'
        "Constraints: selected_models must be a legal C1-C9 member triple; weights>=0 and sum to 1 within 1e-6."
    )


def build_weight_only_prompt(feat: dict, members: list[str]) -> str:
    return (
        "Task: assign non-negative weights that sum to 1 for the FIXED three models below. Do not change the set.\n"
        f"Fixed models: {members}\n"
        f"Market={feat['market']} split={feat['split_name']} origin={feat['origin_date']}\n"
        f"Holiday days in next 28: {feat['holiday_days']}\n"
        f"Discounted single-model MAE on H: {json.dumps({m: feat['single_mae'].get(m) for m in members})}\n"
        f"Stat weights for this set: {feat.get('stat_weights_for_members')}\n"
        f"Dispersion: {feat.get('dispersion')}\n"
        "Output JSON schema:\n"
        '{"selected_models":["id1","id2","id3"],"weights":[w1,w2,w3],'
        '"reason_codes":["..."],"evidence_ids":["..."]}\n'
        "selected_models MUST equal the fixed models in the same order."
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def chat_json(env: dict, user_prompt: str, max_retries: int = 5) -> tuple[dict, dict]:
    url = env["DEEPSEEK_BASE_URL"].rstrip("/") + "/chat/completions"
    body = {
        "model": env["DEEPSEEK_MODEL"],
        "temperature": float(env.get("DEEPSEEK_TEMPERATURE", 0)),
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json",
    }
    delay = 1.0
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - t0
            content = payload["choices"][0]["message"]["content"]
            obj = _extract_json(content)
            meta = {
                "latency_sec": latency,
                "usage": payload.get("usage"),
                "model": payload.get("model", env["DEEPSEEK_MODEL"]),
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "attempt": attempt + 1,
                "raw_content": content,
            }
            return obj, meta
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in {429, 500, 502, 503} or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"DeepSeek failed: {last_err}")


def normalize_weights(weights, n=3):
    import numpy as np
    w = np.asarray(weights, float)
    if w.shape != (n,) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("bad weights")
    s = w.sum()
    if s <= 0:
        raise ValueError("zero weight sum")
    w = w / s
    return [float(x) for x in w]


def validate_select_payload(obj: dict) -> tuple[bool, str, dict]:
    try:
        models = obj.get("selected_models")
        weights = obj.get("weights")
        if not isinstance(models, list) or len(models) != 3:
            return False, "bad_selected_models", {}
        if any(m not in POOL for m in models):
            return False, "model_not_in_pool", {}
        # map to combo id if possible
        ms = set(models)
        combo_id = obj.get("combo_id")
        legal = {cid for cid, mem in COMBOS.items() if cid.startswith("C") and set(mem) == ms}
        if combo_id not in legal:
            combo_id = sorted(legal)[0] if legal else None
        if combo_id is None:
            return False, "not_legal_C1_C9_triple", {}
        # reorder models to combo canonical order
        models = COMBOS[combo_id]
        # reorder weights if needed from original
        # if original order differs, remap
        orig = obj.get("selected_models")
        wmap = {m: float(w) for m, w in zip(orig, weights)}
        w = normalize_weights([wmap[m] for m in models])
        out = {
            "selected_models": models,
            "weights": w,
            "combo_id": combo_id,
            "reason_codes": obj.get("reason_codes") or [],
            "evidence_ids": obj.get("evidence_ids") or [],
        }
        return True, "ok", out
    except Exception as exc:  # noqa: BLE001
        return False, f"exception:{exc}", {}


def validate_weight_payload(obj: dict, members: list[str]) -> tuple[bool, str, dict]:
    try:
        models = obj.get("selected_models")
        weights = obj.get("weights")
        if list(models) != list(members):
            # try remap
            if set(models) != set(members) or len(models) != 3:
                return False, "models_mismatch_fixed_set", {}
            wmap = {m: float(w) for m, w in zip(models, weights)}
            weights = [wmap[m] for m in members]
            models = members
        w = normalize_weights(weights)
        out = {
            "selected_models": list(members),
            "weights": w,
            "reason_codes": obj.get("reason_codes") or [],
            "evidence_ids": obj.get("evidence_ids") or [],
        }
        return True, "ok", out
    except Exception as exc:  # noqa: BLE001
        return False, f"exception:{exc}", {}
