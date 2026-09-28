from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from foundation_models.contracts import AdapterOutput, FoundationAdapter, require_cuda
from foundation_models.adapters._trainable import fit_with_tail_validation, seed_everything, supervised_windows


def _load_patchtst_model(upstream: Path):
    """Load upstream PatchTST without colliding with track1/code/models.py.

    The runner puts its own directory on sys.path, and that directory contains
    the baseline module ``models.py``. A plain ``import models.PatchTST`` then
    binds ``models`` to that file. Loading through a private package name keeps
    the upstream checkout unchanged.
    """
    model_path = upstream / "models" / "PatchTST.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"pinned PatchTST source not found: {model_path}")
    root_name = "patchtst_supervised_upstream"
    if root_name not in sys.modules:
        root = ModuleType(root_name)
        root.__path__ = [str(upstream)]
        sys.modules[root_name] = root
    models_name = f"{root_name}.models"
    if models_name not in sys.modules:
        package = ModuleType(models_name)
        package.__path__ = [str(upstream / "models")]
        package.__package__ = models_name
        sys.modules[models_name] = package
    module_name = f"{models_name}.PatchTST"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {model_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = models_name
    sys.modules[module_name] = module
    upstream_str = str(upstream)
    if upstream_str in sys.path:
        sys.path.remove(upstream_str)
    sys.path.insert(0, upstream_str)
    spec.loader.exec_module(module)
    return module.Model


class PatchTSTAdapter(FoundationAdapter):
    model_name = "patchtst"
    training_mode = "origin_specific_supervised_training"

    def load(self) -> None:
        require_cuda(self.settings.device)
        upstream = Path(self.settings.extra.get("upstream_path", "/opt/upstream/PatchTST_supervised"))
        if not upstream.is_dir():
            raise FileNotFoundError(f"pinned PatchTST source not found: {upstream}")
        try:
            self.model_class = _load_patchtst_model(upstream)
        except (ImportError, FileNotFoundError, AttributeError) as exc:
            raise RuntimeError("could not import PatchTST from the pinned upstream source") from exc

    def _config(self, context: int, horizon: int) -> SimpleNamespace:
        value = self.settings.extra
        return SimpleNamespace(
            enc_in=1, seq_len=context, pred_len=horizon,
            e_layers=int(value.get("e_layers", 3)), n_heads=int(value.get("n_heads", 4)),
            d_model=int(value.get("d_model", 128)), d_ff=int(value.get("d_ff", 256)),
            dropout=float(value.get("dropout", 0.2)), fc_dropout=float(value.get("fc_dropout", 0.2)),
            head_dropout=float(value.get("head_dropout", 0.0)), individual=0,
            patch_len=int(value.get("patch_len", 16)), stride=int(value.get("stride", 8)),
            padding_patch="end", revin=1, affine=0, subtract_last=0,
            decomposition=0, kernel_size=25,
        )

    def predict_batch(self, items, horizon: int) -> list[AdapterOutput]:
        import torch

        results = []
        for item in items:
            seed_everything(self.settings.seed)
            context = int(self.settings.context_length or self.settings.extra.get("context_length", 365))
            x, y, mean, scale = supervised_windows(
                item.history, context, horizon, int(self.settings.extra.get("max_windows", 4096)))
            model = self.model_class(self._config(context, horizon))

            def forward(module, batch_x, _batch_y, _training):
                return module(batch_x)

            training = fit_with_tail_validation(
                model, x, y, device=self.settings.device,
                epochs=int(self.settings.extra.get("epochs", 30)),
                batch_size=self.settings.batch_size,
                learning_rate=float(self.settings.extra.get("learning_rate", 1e-4)),
                patience=int(self.settings.extra.get("patience", 5)), forward=forward)
            model.eval()
            transformed = (np.log1p(item.history[-context:]).astype(np.float32) - mean) / scale
            with torch.no_grad():
                normalized = model(torch.from_numpy(transformed[None, :, None]).to(self.settings.device))
            raw = np.expm1(normalized.detach().cpu().numpy().reshape(-1) * scale + mean)
            checkpoint_dir = self.settings.checkpoint_dir / self.model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{item.task_id}.pt"
            torch.save({"state_dict": model.state_dict(), "config": vars(self._config(context, horizon)),
                        "normalization": {"log1p_mean": mean, "log1p_scale": scale},
                        "training": training}, checkpoint_path)
            results.append(AdapterOutput(
                raw_prediction=raw,
                effective_context_days=context,
                point_source="deterministic_point",
                detail={**training, "zero_shot": False, "covariates": False,
                        "checkpoint_path": str(checkpoint_path)},
            ).validate(horizon))
        return results
