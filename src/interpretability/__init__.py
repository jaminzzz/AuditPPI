"""Model and SAE-feature interpretability algorithms with lazy exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "add_sae_annotations": ("src.interpretability.annotations", "add_sae_annotations"),
    "endpoint_gradient_input_attribution": (
        "src.interpretability.attribution",
        "endpoint_gradient_input_attribution",
    ),
    "ebm_feature_tables": ("src.interpretability.ebm_effects", "ebm_feature_tables"),
    "endpoint_matrix": ("src.interpretability.ebm_effects", "endpoint_matrix"),
    "decoder_attention_weights": (
        "src.interpretability.tabpfn_retrieval",
        "decoder_attention_weights",
    ),
    "embeddings_with_configs": (
        "src.interpretability.tabpfn_retrieval",
        "embeddings_with_configs",
    ),
    "input_overlap_stats": (
        "src.interpretability.tabpfn_retrieval",
        "input_overlap_stats",
    ),
    "l2_normalize": ("src.interpretability.tabpfn_retrieval", "l2_normalize"),
    "top_shared_features": (
        "src.interpretability.tabpfn_retrieval",
        "top_shared_features",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name), attribute)


__all__ = list(_EXPORTS)
