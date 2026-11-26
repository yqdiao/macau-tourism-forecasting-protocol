from __future__ import annotations

import numpy as np

from foundation_models.checkpoints import resolve_snapshot
from foundation_models.contracts import AdapterOutput, FoundationAdapter, SeriesInput, require_cuda


class TimesFM3Adapter(FoundationAdapter):
    model_name = "timesfm_3"
    training_mode = "pretrained_zero_shot"

    def load(self) -> None:
        require_cuda(self.settings.device)
        if not self.settings.checkpoint or not self.settings.checkpoint_revision:
            raise ValueError("TimesFM 3.0 requires a checkpoint and immutable revision")
        local = resolve_snapshot(self.settings.checkpoint, self.settings.checkpoint_revision,
                                 self.settings.cache_dir)
        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
        except ImportError as exc:  # pragma: no cover - model container dependency
            raise RuntimeError("install the pinned TimesFM 3.0 source in this container") from exc
        config = ModelConfig(checkpoint_path=local,
                             per_core_batch_size=self.settings.batch_size,
                             device=self.settings.device)
        self.model = TimesFM3Evaluator(config)

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        contexts = [self.effective_history(item).astype(np.float32) for item in items]
        generated = list(self.model.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=bool(self.settings.extra.get("symmetric_averaging", False)),
        ))
        deciles = np.round(np.arange(0.1, 1.0, 0.1), 1)
        extra = [level for level in self.quantile_levels() if not np.isclose(deciles, level).any()]
        results = []
        for context, value in zip(contexts, generated):
            quantiles = np.asarray(value.quantiles, dtype=float)
            q = {f"q{level:.1f}": quantiles[:, index] for index, level in enumerate(deciles)}
            # The checkpoint only has decile heads; other levels are linear between neighbouring deciles.
            for level in extra:
                q[f"q{level:g}"] = np.array([np.interp(level, deciles, row) for row in quantiles])
            results.append(AdapterOutput(
                raw_prediction=np.asarray(value.forecast),
                effective_context_days=len(context),
                point_source="model_median",
                quantiles=q,
                detail={"zero_shot": True, "covariates": False,
                        "interpolated_quantiles": [float(level) for level in extra]},
            ).validate(horizon))
        return results
