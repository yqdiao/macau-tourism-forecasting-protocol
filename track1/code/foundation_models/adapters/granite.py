from __future__ import annotations

import numpy as np
import pandas as pd

from foundation_models.checkpoints import resolve_snapshot
from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda


class GranitePatchTSTFMAdapter(FoundationAdapter):
    model_name = "granite_patchtst_fm"
    training_mode = "pretrained_zero_shot"

    def load(self) -> None:
        require_cuda(self.settings.device)
        if not self.settings.checkpoint or not self.settings.checkpoint_revision:
            raise ValueError("Granite PatchTST-FM requires a checkpoint and immutable revision")
        local = resolve_snapshot(self.settings.checkpoint, self.settings.checkpoint_revision,
                                 self.settings.cache_dir)
        try:
            from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the pinned granite-tsfm source") from exc
        model = PatchTSTFMForPrediction.from_pretrained(local).to(self.settings.device)
        self.pipeline_class = TimeSeriesForecastingPipeline
        self.model = model

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        levels = self.quantile_levels()
        results = []
        for item in items:
            history = self.effective_history(item)
            frame = pd.DataFrame({
                "date": pd.date_range(end=pd.Timestamp(item.origin), periods=len(history), freq="D"),
                "target": history.astype(np.float32),
            })
            pipeline = self.pipeline_class(
                model=self.model,
                id_columns=[], timestamp_column="date", target_columns=["target"],
                max_context_length=self.model.config.context_length,
                context_length=len(history), prediction_length=horizon,
                impute_method=None, quantile_levels=levels,
                explode_forecasts=True, freq="1d", device=self.settings.device,
                batch_size=self.settings.batch_size,
            )
            output = pipeline(frame)
            median_column = next((c for c in output.columns if str(c).endswith("_q0.5")), None)
            point_column = median_column or "target"
            quantiles = {}
            for level in levels:
                column = next((c for c in output.columns if str(c).endswith(f"_q{level:g}")), None)
                if column:
                    quantiles[f"q{level:g}"] = output[column].to_numpy(float)
            results.append(AdapterOutput(
                raw_prediction=output[point_column].to_numpy(float),
                effective_context_days=len(history),
                point_source="q0.5" if median_column else "model_point",
                quantiles=quantiles,
                detail={"zero_shot": True, "covariates": False},
            ).validate(horizon))
        return results
