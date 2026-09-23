"""Leakage-safe helpers for models trained separately at every forecast origin."""
from __future__ import annotations

import copy
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def supervised_windows(history: np.ndarray, context: int, horizon: int,
                       max_windows: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    transformed = np.log1p(np.asarray(history, dtype=np.float32))
    mean = float(transformed.mean())
    scale = float(transformed.std()) or 1.0
    values = (transformed - mean) / scale
    last_start = len(values) - context - horizon
    if last_start < 0:
        raise ValueError(f"need at least {context + horizon} historical days for training")
    starts = np.arange(last_start + 1)
    if len(starts) > max_windows:
        starts = np.linspace(0, last_start, max_windows, dtype=int)
    x = np.stack([values[start:start + context] for start in starts])[:, :, None]
    y = np.stack([values[start + context:start + context + horizon] for start in starts])[:, :, None]
    return x, y, mean, scale


def fit_with_tail_validation(model, x: np.ndarray, y: np.ndarray, *, device: str,
                             epochs: int, batch_size: int, learning_rate: float,
                             patience: int, forward):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    count = len(x)
    validation_count = max(1, min(count // 5, int(round(count * 0.1))))
    train_count = count - validation_count
    if train_count < 1:
        raise ValueError("not enough historical windows for tail validation")
    train = TensorDataset(torch.from_numpy(x[:train_count]), torch.from_numpy(y[:train_count]))
    valid_x = torch.from_numpy(x[train_count:]).to(device)
    valid_y = torch.from_numpy(y[train_count:]).to(device)
    loader = DataLoader(train, batch_size=min(batch_size, train_count), shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_state, best_loss, stale, epochs_run = None, float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(forward(model, batch_x.to(device), batch_y.to(device), True),
                           batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(forward(model, valid_x, valid_y, False), valid_y).item())
        epochs_run = epoch + 1
        if validation_loss < best_loss - 1e-8:
            best_loss, stale = validation_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return {"epochs_run": epochs_run, "best_validation_mse": best_loss,
            "training_windows": train_count, "validation_windows": validation_count}
