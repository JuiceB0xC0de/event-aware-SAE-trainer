from types import SimpleNamespace

import sae_trainer_rolling as t


def _layer(shared=False, source=None):
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            is_kv_shared_layer=shared,
            kv_shared_layer_index=source,
        )
    )


def test_gemma_kv_dependency_starts_at_shared_layer():
    layers = [_layer() for _ in range(16)]
    layers[15] = _layer(shared=True, source=13)

    assert t._gemma_kv_source_layer(layers, 14) is None
    assert t._gemma_kv_source_layer(layers, 15) == 13


def test_gemma_rolling_float_can_reach_the_full_model_depth():
    assert t._resolve_end_layer(34, n_layers=35, capture="rolling-float") == 34
    assert t._resolve_end_layer(34, n_layers=35, capture="rolling") == 34
