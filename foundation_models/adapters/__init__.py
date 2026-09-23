"""Lazy adapter registry so each model keeps an independent dependency set."""
from __future__ import annotations

from importlib import import_module

from foundation_models.contracts import AdapterSettings, FoundationAdapter

ADAPTER_PATHS = {
    "timesfm_3": "foundation_models.adapters.timesfm3:TimesFM3Adapter",
    "chronos_2": "foundation_models.adapters.chronos2:Chronos2Adapter",
    "lag_llama": "foundation_models.adapters.lag_llama:LagLlamaAdapter",
    "moirai": "foundation_models.adapters.moirai:MoiraiAdapter",
    "granite_patchtst_fm": "foundation_models.adapters.granite:GranitePatchTSTFMAdapter",
    "patchtst": "foundation_models.adapters.patchtst:PatchTSTAdapter",
    "informer": "foundation_models.adapters.informer:InformerAdapter",
}


def create_adapter(name: str, settings: AdapterSettings) -> FoundationAdapter:
    try:
        module_name, class_name = ADAPTER_PATHS[name].split(":")
    except KeyError as exc:
        raise ValueError(f"unknown adapter {name}; choose from {sorted(ADAPTER_PATHS)}") from exc
    adapter_class = getattr(import_module(module_name), class_name)
    return adapter_class(settings)
