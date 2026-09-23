from __future__ import annotations

import tempfile
import unittest
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline import _array_sha256
from foundation_models.contracts import AdapterOutput, SeriesInput
from foundation_models.origin_io import validate_single_origin_input
from prepare_origin_shards import build
from run_foundation_backtest import run


class _FakeAdapter:
    training_mode = "test_only"

    def load(self):
        return None

    def predict_batch(self, items, horizon):
        return [AdapterOutput(np.arange(horizon, dtype=float) - 1.0,
                              len(item.history), "fake_median").validate(horizon)
                for item in items]


class FoundationContractTests(unittest.TestCase):
    def test_series_input_rejects_negative_history(self):
        with self.assertRaises(ValueError):
            SeriesInput("x", date(2023, 1, 1), np.array([1.0] * 27 + [-1.0]))

    def test_adapter_output_requires_exact_horizon(self):
        with self.assertRaises(ValueError):
            AdapterOutput(np.ones(27), 100, "q0.5").validate(28)

    def test_origin_shards_are_physical_and_target_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for fold, origin in (("B01", "2023-01-07"), ("B02", "2023-02-04")):
                history = np.arange(100, dtype=np.int64)
                rows.append({
                    "task_id": fold, "destination": "Macau", "market": "sample",
                    "origin_date": origin, "history_length": "all",
                    "split_name": "recovery_tuning", "fold_id": fold,
                    "history_start": "2022-09-30", "history_end": origin,
                    "training_history_days": len(history), "requested_context_days": None,
                    "effective_context_days": len(history), "horizon_days": 28,
                    "target_start_date": pd.Timestamp(origin) + pd.Timedelta(days=1),
                    "target_end_date": pd.Timestamp(origin) + pd.Timedelta(days=28),
                    "history_sha256": _array_sha256(history), "history": history.tolist(),
                })
            source = root / "task_inputs.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False, engine="pyarrow")
            index = build(source, root / "shards")
            self.assertEqual(index["shard_count"], 2)
            for item in index["shards"]:
                path = root / "shards" / item["relative_path"]
                frame, manifest = validate_single_origin_input(path)
                self.assertEqual(len(frame), 1)
                self.assertFalse(manifest["contains_targets"])
                self.assertNotIn("actual", frame.columns)

    def test_runner_preserves_raw_and_clipped_predictions_without_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = np.arange(100, dtype=np.int64)
            source = root / "task_inputs.parquet"
            pd.DataFrame([{
                "task_id": "B01", "destination": "Macau", "market": "sample",
                "origin_date": "2023-01-07", "history_length": "all",
                "split_name": "recovery_tuning", "fold_id": "B01",
                "history_start": "2022-09-30", "history_end": "2023-01-07",
                "training_history_days": 100, "requested_context_days": None,
                "effective_context_days": 100, "horizon_days": 28,
                "target_start_date": "2023-01-08", "target_end_date": "2023-02-04",
                "history_sha256": _array_sha256(history), "history": history.tolist(),
            }]).to_parquet(source, index=False, engine="pyarrow")
            build(source, root / "shards")
            shard = root / "shards/recovery_tuning/B01_2023-01-07/task_inputs.parquet"
            registry = root / "registry.json"
            registry.write_text(json.dumps({"models": {"fake": {
                "repository": "https://example.invalid/fake.git",
                "source_revision": "a" * 40, "checkpoint": None,
                "checkpoint_revision": None, "adapter_status": "implemented",
                "runtime": {"context_length": None, "extra": {}},
            }}}), encoding="utf-8")
            with patch("run_foundation_backtest.create_adapter", return_value=_FakeAdapter()):
                manifest = run(model_name="fake", inputs=shard, output=root / "output",
                               registry=registry, context_length=None, batch_size=1,
                               num_samples=10, device="cpu", seed=42,
                               history_lengths=("all",))
            predictions = pd.read_parquet(root / "output/predictions.parquet")
            self.assertEqual(len(predictions), 28)
            self.assertEqual(predictions.iloc[0]["prediction_raw"], -1.0)
            self.assertEqual(predictions.iloc[0]["prediction"], 0.0)
            self.assertTrue(predictions.iloc[0]["was_clipped"])
            self.assertFalse(manifest["target_artifact_accessed"])


if __name__ == "__main__":
    unittest.main()
