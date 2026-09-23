from __future__ import annotations

import numpy as np
import pandas as pd

from foundation_models.checkpoints import resolve_snapshot
from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda


class MoiraiAdapter(FoundationAdapter):
    model_name = "moirai"
    training_mode = "pretrained_zero_shot"

    def load(self) -> None:
        require_cuda(self.settings.device)
        if not self.settings.checkpoint or not self.settings.checkpoint_revision:
            raise ValueError("Moirai requires a checkpoint and immutable revision")
        local = resolve_snapshot(self.settings.checkpoint, self.settings.checkpoint_revision,
                                 self.settings.cache_dir)
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("this adapter requires the pinned Uni2TS source with Moirai 2") from exc
        self.forecast_class = Moirai2Forecast
        self.module = Moirai2Module.from_pretrained(local)

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        from gluonts.dataset.common import ListDataset

        results = []
        for item in items:
            history = self.effective_history(item)
            context_length = len(history)
            model = self.forecast_class(
                module=self.module,
                prediction_length=horizon,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            predictor = model.create_predictor(batch_size=1, device=self.settings.device)
            start = pd.Period(item.origin, freq="D") - (len(history) - 1)
            dataset = ListDataset([{"item_id": item.task_id, "start": start,
                                    "target": history}], freq="D")
            forecast = next(iter(predictor.predict(dataset)))
            quantiles = {f"q{q}": np.asarray(forecast.quantile(q), dtype=float)
                         for q in (0.1, 0.5, 0.9)}
            results.append(AdapterOutput(
                raw_prediction=quantiles["q0.5"],
                effective_context_days=context_length,
                point_source="q0.5",
                quantiles=quantiles,
                detail={"zero_shot": True, "covariates": False},
            ).validate(horizon))
        return results
