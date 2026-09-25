"""Unified 28-day point-forecast adapters; all receive historical counts only."""
from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the locally installed wheels when running on this computer. A normal
# Python environment can instead install requirements.txt and ignore this path.
LOCAL_DEPS = Path(__file__).resolve().parents[2] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

TABULAR_MODELS = (
    "ridge", "linear_regression", "random_forest", "gradient_boosting",
    "svr", "lasso", "elasticnet", "parrot_random_forest_800", "lightgbm",
)
MODEL_NAMES = ("seasonal_naive", "ets", "sarima", *TABULAR_MODELS, "transformer")


@dataclass(frozen=True)
class ForecastResult:
    model: str
    raw_prediction: np.ndarray
    prediction: np.ndarray
    fit_seconds: float
    inference_seconds: float
    detail: dict


def _history(values) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 90 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Provide at least 90 historical daily nonnegative counts")
    return y


def _result(name, forecast, fit_seconds, inference_seconds, detail=None):
    p = np.asarray(forecast, dtype=float).reshape(-1)
    if not np.isfinite(p).all():
        raise ValueError(f"{name}: forecast contains nonfinite values")
    return ForecastResult(
        name,
        p.copy(),
        np.maximum(0.0, p),
        float(fit_seconds),
        float(inference_seconds),
        detail or {},
    )


def seasonal_naive(history, horizon=28, *, config: dict | None = None) -> ForecastResult:
    y = _history(history)
    if config and config != {"period": 7}:
        raise ValueError("seasonal_naive only supports period=7")
    start = time.perf_counter()
    # For h > 7, repeat the last OBSERVED week, never a week inside the test block.
    p = np.resize(y[-7:], horizon)
    return _result("seasonal_naive", p, 0, time.perf_counter() - start,
                   {"period": 7, "template": "last observed week", "effective_context_days": 7})


def ets(history, horizon=28, *, config: dict | None = None) -> ForecastResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = _history(history)
    params = {"transform": "log1p", "trend": None, "damped_trend": False,
              "seasonal": "add", "seasonal_periods": 7}
    params.update(config or {})
    if params["transform"] != "log1p" or params["seasonal_periods"] != 7:
        raise ValueError("ETS requires log1p and weekly seasonality")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(np.log1p(y), trend=params["trend"],
                                   damped_trend=params["damped_trend"],
                                   seasonal=params["seasonal"],
                                   seasonal_periods=7,
                                   initialization_method="estimated").fit(optimized=True)
    fitted = time.perf_counter()
    p = np.expm1(np.asarray(fit.forecast(horizon), dtype=float))
    return _result("ets", p, fitted - start, time.perf_counter() - fitted,
                   {"specification": params, "period": 7,
                    "effective_context_days": len(y)})


def sarima(history, horizon=28, *, config: dict | None = None) -> ForecastResult:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = _history(history)
    params = {"transform": "log1p", "order": [1, 0, 0],
              "seasonal_order": [1, 0, 0, 7], "trend": "c"}
    params.update(config or {})
    if params["transform"] != "log1p" or params["seasonal_order"][-1] != 7:
        raise ValueError("SARIMA requires log1p and weekly seasonality")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(np.log1p(y), order=tuple(params["order"]),
                      seasonal_order=tuple(params["seasonal_order"]),
                      trend=params["trend"], enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False, maxiter=50)
    fitted = time.perf_counter()
    p = np.expm1(np.asarray(fit.forecast(horizon), dtype=float))
    return _result("sarima", p, fitted - start, time.perf_counter() - fitted,
                   {"order": params["order"], "seasonal_order": params["seasonal_order"],
                    "converged": bool(fit.mle_retvals.get("converged", False)),
                    "effective_context_days": len(y)})


def _features(log_values: np.ndarray, t: int, h: int) -> np.ndarray:
    """Only current/past log arrivals; no future observations or outside market."""
    lag = [log_values[t - n + 1] for n in range(1, 29)]
    stats = []
    for window in (7, 14, 28, 56):
        part = log_values[t - window + 1:t + 1]
        stats.extend((part.mean(), part.std()))
    offset = (7 - h % 7) % 7
    weekly = [log_values[t - offset - 7 * k] for k in range(4)]
    return np.asarray([*lag, *stats, np.mean(weekly)], dtype=np.float32)


def _build_regressor(name: str, seed: int, config: dict | None = None):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    config = config or {}
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=config.get("alpha", 10.0)))
    if name == "linear_regression":
        return make_pipeline(StandardScaler(), LinearRegression(
            fit_intercept=config.get("fit_intercept", True)))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=config.get("n_estimators", 100),
                                     max_depth=config.get("max_depth", 10),
                                     min_samples_split=config.get("min_samples_split", 5),
                                     random_state=seed, n_jobs=2)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=config.get("n_estimators", 100),
                                         max_depth=config.get("max_depth", 6),
                                         learning_rate=config.get("learning_rate", 0.05),
                                         random_state=seed)
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(
            kernel=config.get("kernel", "rbf"), C=config.get("C", 0.1),
            epsilon=config.get("epsilon", 0.01)))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(
            alpha=config.get("alpha", 0.001), max_iter=10000, random_state=seed))
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(
            alpha=config.get("alpha", 0.001), l1_ratio=config.get("l1_ratio", 0.3),
            max_iter=10000, random_state=seed))
    if name == "parrot_random_forest_800":
        return RandomForestRegressor(n_estimators=config.get("n_estimators", 800),
                                     max_depth=config.get("max_depth"),
                                     random_state=seed, n_jobs=2)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="regression",
                                 n_estimators=config.get("n_estimators", 100),
                                 learning_rate=config.get("learning_rate", 0.05),
                                 num_leaves=config.get("num_leaves", 31),
                                 max_depth=config.get("max_depth", 8),
                                 min_child_samples=10, colsample_bytree=0.75,
                                 subsample=0.75, subsample_freq=1,
                                 random_state=seed, n_jobs=2, verbosity=-1)
    raise ValueError(f"Unknown tabular model: {name}")


def tabular_direct(name: str, history, horizon=28, seed=42,
                   *, config: dict | None = None, prepared=None) -> ForecastResult:
    if name not in TABULAR_MODELS:
        raise ValueError(f"Choose one of: {', '.join(TABULAR_MODELS)}")
    y = _history(history)
    if len(y) < 56 + horizon + 20:
        raise ValueError(f"{name}: too little training history for {horizon} direct horizons")
    z = np.log1p(y)
    predictions = []
    fit_seconds = inference_seconds = 0.0
    training_sets = prepared if prepared is not None else prepare_tabular_training_sets(y, horizon)
    for h in range(1, horizon + 1):
        t0 = time.perf_counter()
        x, target, inference_x = training_sets[h - 1]
        model = _build_regressor(name, seed, config)
        model.fit(x, target)
        fit_seconds += time.perf_counter() - t0
        t1 = time.perf_counter()
        prediction = float(model.predict(inference_x)[0])
        inference_seconds += time.perf_counter() - t1
        predictions.append(math.expm1(min(30.0, prediction)))
    return _result(name, predictions, fit_seconds, inference_seconds,
                   {"strategy": "one market, one direct regressor per horizon",
                    "horizon_models": horizon, "history_features_only": True,
                    "lookback_days": 56, "effective_context_days": 56,
                    "training_history_days": len(y)})


def prepare_tabular_training_sets(history, horizon=28):
    """Build each direct horizon's features once for reuse across candidates."""
    z = np.log1p(_history(history))
    out = []
    for h in range(1, horizon + 1):
        t = np.arange(55, len(z) - h)
        lag = z[t[:, None] - np.arange(28)]
        stats = []
        for window in (7, 14, 28, 56):
            blocks = z[t[:, None] - np.arange(window - 1, -1, -1)]
            stats.extend((blocks.mean(axis=1), blocks.std(axis=1)))
        offset = (7 - h % 7) % 7
        weekly = z[t[:, None] - offset - 7 * np.arange(4)].mean(axis=1)
        x = np.column_stack((lag, *stats, weekly)).astype(np.float32)
        target = z[t + h]
        inference_x = _features(z, len(z) - 1, h).reshape(1, -1)
        out.append((x, target, inference_x))
    return out


def transformer(history, horizon=28, seed=42, epochs=8, lookback=56,
                *, config: dict | None = None) -> ForecastResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    params = {"lookback": lookback, "epochs": epochs, "learning_rate": 0.001,
              "d_model": 32}
    params.update(config or {})
    lookback, epochs, d_model = params["lookback"], params["epochs"], params["d_model"]
    y = _history(history)
    if len(y) < lookback + horizon + 20:
        raise ValueError("Transformer needs more historical days")
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    logs = np.log1p(y)
    mean, scale = float(logs.mean()), float(logs.std()) or 1.0
    standardized = (logs - mean) / scale
    x, target = [], []
    for t in range(lookback - 1, len(y) - horizon):
        x.append(standardized[t - lookback + 1:t + 1])
        target.append(standardized[t + 1:t + horizon + 1])
    data = TensorDataset(torch.tensor(np.asarray(x)[:, :, None], dtype=torch.float32),
                         torch.tensor(np.asarray(target), dtype=torch.float32))
    loader = DataLoader(data, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    class DemandTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, d_model)
            positions = torch.arange(lookback, dtype=torch.float32).unsqueeze(1)
            freqs = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / d_model))
            encoding = torch.zeros(lookback, d_model)
            encoding[:, 0::2] = torch.sin(positions * freqs)
            encoding[:, 1::2] = torch.cos(positions * freqs)
            self.register_buffer("position", encoding.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                               dim_feedforward=64, dropout=0.1,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d_model, horizon)

        def forward(self, history_window):
            return self.head(self.encoder(self.projection(history_window) + self.position)[:, -1, :])

    net = DemandTransformer()
    optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate"])
    loss_fn = nn.MSELoss()
    net.train()
    epoch_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(batch), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        epoch_losses.append(total_loss / len(data))
    fitted = time.perf_counter()
    net.eval()
    with torch.no_grad():
        inp = torch.tensor(standardized[-lookback:][None, :, None], dtype=torch.float32)
        pred = net(inp).numpy()[0]
    inference = time.perf_counter() - fitted
    return _result("transformer", np.expm1(np.clip(pred * scale + mean, -20, 30)),
                   fitted - t0, inference,
                   {"lookback": lookback, "epochs": epochs, "training_windows": len(data),
                    "learning_rate": params["learning_rate"], "d_model": d_model,
                    "epoch_train_loss": epoch_losses,
                    "effective_context_days": lookback, "training_history_days": len(y),
                    "architecture": "local encoder regression, not a pretrained foundation model"})


def forecast(name: str, history, horizon: int = 28, seed: int = 42,
             *, config: dict | None = None, prepared=None) -> ForecastResult:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name}. Available: {', '.join(MODEL_NAMES)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if name == "seasonal_naive":
        return seasonal_naive(history, horizon, config=config)
    if name == "ets":
        return ets(history, horizon, config=config)
    if name == "sarima":
        return sarima(history, horizon, config=config)
    if name == "transformer":
        return transformer(history, horizon, seed, config=config)
    return tabular_direct(name, history, horizon, seed, config=config, prepared=prepared)
