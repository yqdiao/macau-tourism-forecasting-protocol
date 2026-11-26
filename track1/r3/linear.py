"""NLinear-style models with purged validation and optional full refit."""
from __future__ import annotations

import time

import numpy as np

from data import OriginBundle, seasonal_naive
from features import week4_baseline
from scale import prefix_scales

LOOKBACK = 56
HORIZON = 28


def _module(kind: str):
    import torch
    from torch import nn

    class NLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(LOOKBACK, HORIZON)

        def forward(self, x):
            last = x[:, -1:]
            return self.linear(x - last) + last

    class ScaledResidual(nn.Module):
        def __init__(self, calendar: bool):
            super().__init__()
            self.linear = nn.Linear(LOOKBACK, HORIZON)
            self.calendar = nn.Linear(21, 1, bias=False) if calendar else None

        def forward(self, x, calendar=None):
            base = self.linear(x)
            if self.calendar is not None:
                base = base + (calendar * self.calendar.weight.view(1, 1, 21)).sum(-1)
            return base

    if kind == "nlinear":
        return NLinear()
    if kind == "scaled":
        return ScaledResidual(False)
    if kind == "scaled_calendar":
        return ScaledResidual(True)
    raise ValueError(kind)


def _windows(z: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = z[anchors[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :]]
    y = z[anchors[:, None] + np.arange(1, HORIZON + 1)[None, :]]
    return x.astype(np.float32), y.astype(np.float32)


def _calendar_batch(calendar, market: str, start, anchors: np.ndarray) -> np.ndarray:
    """(n, 28, 21) weekday, month, macau, mainland."""
    import pandas as pd
    n = len(anchors)
    h = np.tile(np.arange(1, 29), n)
    t = np.repeat(anchors, 28)
    ords = np.datetime64(start) + (t + h).astype("timedelta64[D]")
    days = pd.DatetimeIndex(ords.astype("datetime64[D]"))
    python_days = [d.date() for d in days]
    dow = np.eye(7, dtype=np.float32)[days.dayofweek.to_numpy()]
    month = np.eye(12, dtype=np.float32)[days.month.to_numpy() - 1]
    macau = np.fromiter((d in calendar.macau for d in python_days), dtype=np.float32, count=len(python_days))
    mainland = np.fromiter((d in calendar.mainland for d in python_days), dtype=np.float32, count=len(python_days))
    feat = np.column_stack([dow, month, macau, mainland]).reshape(n, 28, 21)
    return feat


def fit_linear(bundle: OriginBundle, spec: dict, calendar, *, seed: int, threads: int = 2,
               smoke: bool = False) -> dict:
    import copy
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = bundle.values.shape[1]
    o_idx = n - 1
    v = [o_idx - 84, o_idx - 56, o_idx - 28]
    if v[0] < LOOKBACK:
        return _all_fallback(bundle, "history_shorter_than_validation")
    select_last = v[0] - HORIZON
    select_anchors = np.arange(LOOKBACK - 1, select_last + 1)
    refit_anchors = np.arange(LOOKBACK - 1, (o_idx - HORIZON) + 1)
    if int(select_anchors[-1] + HORIZON) != int(v[0]) or int(refit_anchors[-1] + HORIZON) != int(o_idx):
        raise RuntimeError("validation or refit label boundary is wrong")
    kind = spec["kind"]
    max_epochs = 2 if smoke else 50
    patience = 5
    batch_size = 256
    started = time.perf_counter()
    sx, sy, sw, sc = _collect(bundle, select_anchors, kind, calendar)
    model = _module(kind)
    n_params = sum(p.numel() for p in model.parameters())
    expected = 1617 if kind == "scaled_calendar" else 1596
    if n_params != expected:
        raise RuntimeError(f"parameter count {n_params} != {expected}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_loss, best_epoch, stale, epochs_run = None, float("inf"), 0, 0, 0
    history = []
    vx, vy, vw, vc, vq, vy_raw, vb = _validation(bundle, v, kind, calendar)
    for epoch in range(1, max_epochs + 1):
        _train_epoch(model, optimizer, sx, sy, sw, sc, seed, epoch, batch_size, kind)
        epochs_run = epoch
        current = _selection_loss(model, vx, vy, vw, vc, vq, vy_raw, vb, kind)
        history.append({"epoch": epoch, "selection_loss": current})
        if current < best_loss - 1e-8:
            best_loss, best_epoch, stale = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    select_seconds = time.perf_counter() - started
    refit_seconds = 0.0
    if spec["refit"]:
        started = time.perf_counter()
        torch.manual_seed(seed)
        model = _module(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rx, ry, rw, rc = _collect(bundle, refit_anchors, kind, calendar)
        for epoch in range(1, best_epoch + 1):
            _train_epoch(model, optimizer, rx, ry, rw, rc, seed, epoch, batch_size, kind)
        refit_seconds = time.perf_counter() - started
    else:
        model.load_state_dict(best_state)
    predictions, statuses = {}, {}
    model.eval()
    with torch.no_grad():
        for i, market in enumerate(bundle.markets):
            try:
                raw = _forecast_one(model, bundle, i, kind, calendar)
                if not np.isfinite(raw).all():
                    raise ValueError("non-finite")
                predictions[market] = raw
                statuses[market] = "native"
            except Exception as exc:  # noqa: BLE001
                predictions[market] = seasonal_naive(bundle.values[i])
                statuses[market] = f"fallback:{type(exc).__name__}"
    return {
        "predictions": predictions, "status": statuses, "fit_seconds": select_seconds + refit_seconds,
        "select_seconds": select_seconds, "refit_seconds": refit_seconds, "inference_seconds": 0.0,
        "best_epoch": best_epoch, "epochs_run": epochs_run, "n_params": n_params,
        "selection_anchors": int(len(select_anchors)), "refit_anchors": int(len(refit_anchors) if spec["refit"] else 0),
        "epoch_history": history, "best_loss": best_loss,
        "final_label_index": int(refit_anchors[-1] + HORIZON if spec["refit"] else select_anchors[-1] + HORIZON),
    }


def _all_fallback(bundle, reason):
    return {
        "predictions": {m: seasonal_naive(bundle.values[i]) for i, m in enumerate(bundle.markets)},
        "status": {m: f"fallback:{reason}" for m in bundle.markets},
        "fit_seconds": 0.0, "select_seconds": 0.0, "refit_seconds": 0.0, "inference_seconds": 0.0,
        "best_epoch": 0, "epochs_run": 0, "n_params": 0, "selection_anchors": 0, "refit_anchors": 0,
        "epoch_history": [], "best_loss": None, "final_label_index": None,
    }


def _collect(bundle, anchors, kind, calendar):
    xs, ys, ws, cs = [], [], [], []
    for i, market in enumerate(bundle.markets):
        y = bundle.values[i]
        q = prefix_scales(y)
        if kind == "nlinear":
            x, target = _windows(np.log1p(y).astype(np.float32), anchors)
            w = np.ones(len(anchors), dtype=np.float32)
            c = None
        else:
            mu = y[anchors[:, None] - np.arange(27, -1, -1)[None, :]].mean(axis=1)
            qq = q[anchors]
            ok = np.isfinite(qq) & (qq > 0)
            use = anchors[ok]
            mu, qq = mu[ok], qq[ok]
            hist = y[use[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :]]
            x = ((hist - mu[:, None]) / qq[:, None]).astype(np.float32)
            base = week4_baseline(y, use)
            future = y[use[:, None] + np.arange(1, 29)[None, :]]
            target = ((future - base.reshape(len(use), 28)) / qq[:, None]).astype(np.float32)
            w = np.full(len(use), 1.0 / max(ok.sum(), 1), dtype=np.float32)
            c = _calendar_batch(calendar, market, bundle.start, use) if kind == "scaled_calendar" else None
        xs.append(x)
        ys.append(target)
        ws.append(w)
        if c is not None:
            cs.append(c)
    w = np.concatenate(ws).astype(np.float32)
    w = w / w.mean()
    c = np.concatenate(cs) if cs else None
    return np.vstack(xs), np.vstack(ys), w, c


def _validation(bundle, origins, kind, calendar):
    anchors = np.array(origins)
    return (*_collect(bundle, anchors, kind, calendar), *_valid_scale(bundle, anchors, kind))


def _valid_scale(bundle, anchors, kind):
    qs, raws, bases = [], [], []
    for y in bundle.values:
        q = prefix_scales(y)[anchors]
        qs.append(np.repeat(q, 28))
        raws.append(y[anchors[:, None] + np.arange(1, 29)[None, :]].reshape(-1))
        if kind == "nlinear":
            bases.append(np.zeros(len(anchors) * 28))
        else:
            bases.append(week4_baseline(y, anchors))
    return np.concatenate(qs), np.concatenate(raws), np.concatenate(bases)


def _train_epoch(model, optimizer, x, y, w, c, seed, epoch, batch_size, kind):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    model.train()
    tensors = [torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)]
    if c is not None:
        tensors.append(torch.from_numpy(c))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + epoch))
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch[0]) if c is None else model(batch[0], batch[3])
        err = (pred - batch[1]).abs() if kind != "nlinear" else (pred - batch[1]).pow(2)
        loss = (batch[2] * err.mean(dim=1)).mean()
        loss.backward()
        optimizer.step()


def _selection_loss(model, x, y, w, c, q, y_raw, base, kind) -> float:
    import torch
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)) if c is None else model(torch.from_numpy(x), torch.from_numpy(c))
        pred = pred.numpy()
    if kind == "nlinear":
        err = (pred - y.reshape(len(pred), 28)) ** 2
        # equal weight across markets and the anchors in this batch, which are the 3 origins stacked by market
        per = err.mean(axis=1)
        return float(per.mean())
    raw = base.reshape(len(pred), 28) + q.reshape(len(pred), 28) * pred
    actual = y_raw.reshape(len(pred), 28)
    qq = q.reshape(len(pred), 28)[:, 0]
    mase = np.mean(np.abs(raw - actual), axis=1) / qq
    return float(np.mean(mase))


def _forecast_one(model, bundle, i, kind, calendar):
    import torch
    y = bundle.values[i]
    anchor = np.array([len(y) - 1])
    if kind == "nlinear":
        z = np.log1p(y).astype(np.float32)
        window = torch.from_numpy(z[-LOOKBACK:][None, :])
        return np.expm1(model(window).numpy().reshape(-1))
    mu = y[-28:].mean()
    q = prefix_scales(y)[-1]
    x = torch.from_numpy(((y[-LOOKBACK:] - mu) / q).astype(np.float32)[None, :])
    c = None
    if kind == "scaled_calendar":
        c = torch.from_numpy(_calendar_batch(calendar, bundle.markets[i], bundle.start, anchor))
    u = model(x, c).numpy().reshape(-1) if c is not None else model(x).numpy().reshape(-1)
    return week4_baseline(y, anchor) + q * u
