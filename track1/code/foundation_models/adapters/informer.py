from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda
from foundation_models.adapters._trainable import (
    encode_context, fit_with_tail_validation, seed_everything, supervised_windows,
)


class InformerAdapter(FoundationAdapter):
    model_name = "informer"
    training_mode = "origin_specific_supervised_training"

    def load(self) -> None:
        require_cuda(self.settings.device)
        upstream = Path(self.settings.extra.get("upstream_path", "/opt/upstream"))
        if not upstream.is_dir():
            raise FileNotFoundError(f"pinned Informer source not found: {upstream}")
        sys.path.insert(0, str(upstream))
        try:
            from models.model import Informer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("could not import Informer from the pinned upstream source") from exc
        self.model_class = Informer

    def _model(self, context: int, label: int, horizon: int):
        import torch
        value = self.settings.extra
        return self.model_class(
            enc_in=1, dec_in=1, c_out=1, seq_len=context, label_len=label,
            out_len=horizon, factor=int(value.get("factor", 5)),
            d_model=int(value.get("d_model", 128)), n_heads=int(value.get("n_heads", 4)),
            e_layers=int(value.get("e_layers", 2)), d_layers=int(value.get("d_layers", 1)),
            d_ff=int(value.get("d_ff", 256)), dropout=float(value.get("dropout", 0.1)),
            attn="prob", embed="fixed", freq="d", activation="gelu",
            output_attention=False, distil=True, mix=True,
            device=torch.device(self.settings.device),
        )

    @staticmethod
    def _forward(module, batch_x, batch_y, training: bool, label: int, horizon: int):
        import torch
        decoder = torch.cat([batch_x[:, -label:, :],
                             torch.zeros((len(batch_x), horizon, 1), device=batch_x.device)], dim=1)
        # Constant valid calendar marks preserve the requested univariate-only experiment.
        encoder_marks = torch.zeros((len(batch_x), batch_x.shape[1], 4), device=batch_x.device)
        decoder_marks = torch.zeros((len(batch_x), label + horizon, 4), device=batch_x.device)
        return module(batch_x, encoder_marks, decoder, decoder_marks)

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        import torch

        results = []
        for item in items:
            seed_everything(self.settings.seed)
            context = int(self.settings.context_length or self.settings.extra.get("context_length", 365))
            label = int(self.settings.extra.get("label_length", min(28, context // 2)))
            normalization = str(self.settings.extra.get("normalization", "global"))
            x, y, mean, scale = supervised_windows(
                item.history, context, horizon, int(self.settings.extra.get("max_windows", 4096)),
                normalization)
            model = self._model(context, label, horizon)

            def forward(module, batch_x, batch_y, training):
                return self._forward(module, batch_x, batch_y, training, label, horizon)

            training = fit_with_tail_validation(
                model, x, y, device=self.settings.device,
                epochs=int(self.settings.extra.get("epochs", 30)),
                batch_size=self.settings.batch_size,
                learning_rate=float(self.settings.extra.get("learning_rate", 1e-4)),
                patience=int(self.settings.extra.get("patience", 5)), forward=forward,
                seed=self.settings.seed)
            model.eval()
            transformed, center = encode_context(item.history, context, mean, scale, normalization)
            batch_x = torch.from_numpy(transformed[None, :, None]).to(self.settings.device)
            with torch.no_grad():
                normalized = self._forward(model, batch_x, torch.empty(0), False, label, horizon)
            raw = np.expm1(normalized.detach().cpu().numpy().reshape(-1) * scale + center)
            checkpoint_dir = self.settings.checkpoint_dir / self.model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{item.task_id}.pt"
            torch.save({"state_dict": model.state_dict(),
                        "config": {"context_length": context, "label_length": label,
                                   "prediction_length": horizon},
                        "normalization": {"mode": normalization, "log1p_mean": mean,
                                          "log1p_scale": scale, "context_center": center},
                        "training": training}, checkpoint_path)
            results.append(AdapterOutput(
                raw_prediction=raw,
                effective_context_days=context,
                point_source="deterministic_point",
                detail={**training, "zero_shot": False, "covariates": False,
                        "normalization": normalization,
                        "constant_time_marks": True, "checkpoint_path": str(checkpoint_path)},
            ).validate(horizon))
        return results
