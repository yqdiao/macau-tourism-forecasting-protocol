from __future__ import annotations

import numpy as np
import pandas as pd

from foundation_models.checkpoints import resolve_snapshot
from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda


class Chronos2Adapter(FoundationAdapter):
    model_name = "chronos_2"
    training_mode = "pretrained_zero_shot"

    def load(self) -> None:
        require_cuda(self.settings.device)
        if not self.settings.checkpoint or not self.settings.checkpoint_revision:
            raise ValueError("Chronos-2 requires a checkpoint and immutable revision")
        local = resolve_snapshot(self.settings.checkpoint, self.settings.checkpoint_revision,
                                 self.settings.cache_dir)
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the pinned chronos-forecasting source") from exc
        self.model = Chronos2Pipeline.from_pretrained(local, device_map=self.settings.device)

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        frames, lengths = [], []
        for item in items:
            history = self.effective_history(item)
            dates = pd.date_range(end=pd.Timestamp(item.origin), periods=len(history), freq="D")
            frames.append(pd.DataFrame({"item_id": item.task_id, "timestamp": dates,
                                        "target": history.astype(np.float32)}))
            lengths.append(len(history))
        levels = self.quantile_levels()
        forecast = self.model.predict_df(
            pd.concat(frames, ignore_index=True),
            prediction_length=horizon,
            quantile_levels=levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=self.settings.batch_size,
            context_length=self.settings.context_length,
            cross_learning=False,
            freq="D",
        )
        results = []
        for item, length in zip(items, lengths):
            rows = forecast[forecast["item_id"] == item.task_id].sort_values("timestamp")
            results.append(AdapterOutput(
                raw_prediction=rows["0.5"].to_numpy(float),
                effective_context_days=length,
                point_source="q0.5",
                quantiles={f"q{q:g}": rows[f"{q:g}"].to_numpy(float) for q in levels},
                detail={"zero_shot": True, "cross_learning": False, "covariates": False},
            ).validate(horizon))
        return results
