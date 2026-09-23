from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_models.checkpoints import resolve_snapshot
from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda


class LagLlamaAdapter(FoundationAdapter):
    model_name = "lag_llama"
    training_mode = "pretrained_zero_shot"

    def load(self) -> None:
        require_cuda(self.settings.device)
        if not self.settings.checkpoint or not self.settings.checkpoint_revision:
            raise ValueError("Lag-Llama requires a checkpoint and immutable revision")
        local = Path(resolve_snapshot(self.settings.checkpoint, self.settings.checkpoint_revision,
                                      self.settings.cache_dir))
        self.checkpoint_path = local / str(self.settings.extra.get("checkpoint_file", "lag-llama.ckpt"))
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        try:
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the pinned Lag-Llama and GluonTS dependencies") from exc
        # The official checkpoint contains Python objects. It is loaded only after the
        # immutable repository revision has been verified by snapshot_download.
        checkpoint = torch.load(self.checkpoint_path, map_location=self.settings.device,
                                weights_only=False)
        self.model_kwargs = checkpoint["hyper_parameters"]["model_kwargs"]
        self.estimator_class = LagLlamaEstimator
        self.torch = torch

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        from gluonts.dataset.common import ListDataset

        results = []
        signature = inspect.signature(self.estimator_class.__init__)
        for item in items:
            history = self.effective_history(item)
            context_length = len(history)
            requested = dict(self.model_kwargs)
            requested.update({
                "ckpt_path": str(self.checkpoint_path),
                "prediction_length": horizon,
                "context_length": context_length,
                "num_parallel_samples": self.settings.num_samples,
                "device": self.torch.device(self.settings.device),
                "nonnegative_pred_samples": False,
            })
            kwargs = {key: value for key, value in requested.items()
                      if key in signature.parameters}
            estimator = self.estimator_class(**kwargs)
            module = estimator.create_lightning_module()
            transform = estimator.create_transformation()
            predictor = estimator.create_predictor(transform, module)
            start = pd.Period(item.origin, freq="D") - (len(history) - 1)
            dataset = ListDataset([{"item_id": item.task_id, "start": start,
                                    "target": history}], freq="D")
            forecast = next(iter(predictor.predict(dataset, num_samples=self.settings.num_samples)))
            samples = np.asarray(forecast.samples, dtype=float)
            raw = np.median(samples, axis=0)
            results.append(AdapterOutput(
                raw_prediction=raw,
                effective_context_days=context_length,
                point_source="sample_median",
                quantiles={f"q{q}": np.quantile(samples, q, axis=0)
                           for q in (0.1, 0.5, 0.9)},
                detail={"zero_shot": True, "num_samples": samples.shape[0],
                        "covariates": False},
            ).validate(horizon))
        return results
