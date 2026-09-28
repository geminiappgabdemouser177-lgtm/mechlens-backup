"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.125 (canonical Command-R is 0.0625; tiny diverges so
    regression tests catch silent-fallback bugs in the passthrough)

NOTE: The tiny model has use_qk_norm=False, so QK-norm is not exercised here.
These tests cover the unprocessed tiny-model path, not production checkpoints or
full compatibility-mode processing. Longer-input parity may expose adapter bugs;
do not weaken the assertions to make an implementation mismatch pass.

Run from the fork's configured environment:
    uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
pytestmark = pytest.mark.slow
LOGIT_ATOL = 1e-4
ACTIVATION_ATOL = 1e-5
LONG_SEQUENCE_LENGTH = 128
LONG_PROMPT = (
    "At dawn, the research team checked the instruments beside the harbor. "
    "Mira recorded seven measurements, then compared them with yesterday's log. "
    "The first sensor stayed steady, but the second changed after the door opened. "
    "Why did the readings disagree? They repeated the experiment with the door closed, "
    "swapped the sensors, and wrote down the sequence before drawing a conclusion. "
    "Later, a colleague reviewed the notes and asked whether temperature, timing, "
    "or calibration could explain the difference. Every observation remained in the report. "
) * 4


def _load_cohere_bridge() -> TransformerBridge:
    """Match the independent HF reference's precision and evaluation mode."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.eval()
    bridge.original_model.eval()
    assert bridge.original_model.config._attn_implementation == "eager"
    return bridge


@pytest.fixture(scope="module", params=["short", "long"])
def cohere_tokens(request: pytest.FixtureRequest, cohere_bridge: TransformerBridge) -> torch.Tensor:
    """Use identical IDs on both sides; retain the old case plus 128 text tokens."""
    if request.param == "short":
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    tokens = cohere_bridge.tokenizer(
        LONG_PROMPT, return_tensors="pt", add_special_tokens=True, truncation=False
    )["input_ids"]
    assert tokens.shape[1] >= LONG_SEQUENCE_LENGTH, "Long prompt unexpectedly tokenized too short"
    max_positions = cohere_bridge.original_model.config.max_position_embeddings
    assert max_positions >= LONG_SEQUENCE_LENGTH, "Checkpoint context is too short for this test"
    return tokens[:, :LONG_SEQUENCE_LENGTH].contiguous()


def _assert_max_abs_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, label: str) -> None:
    """Enforce absolute error without a relative-tolerance escape hatch."""
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype == torch.float32, f"{label}: expected float32"
    assert torch.isfinite(actual).all().item(), f"{label}: bridge contains non-finite values"
    assert torch.isfinite(expected).all().item(), f"{label}: HF contains non-finite values"
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < atol, f"{label}: max absolute difference {max_diff:.8g} >= {atol}"


def _capture_activation(name: str, activations: dict[str, torch.Tensor]) -> Any:
    """Copy HF outputs so later operations cannot mutate the reference."""

    def capture(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        assert isinstance(output, torch.Tensor), f"{name}: expected a tensor output"
        assert name not in activations, f"{name}: unexpectedly ran more than once"
        activations[name] = output.detach().clone()

    return capture


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return _load_cohere_bridge()


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically. We disable all ProcessWeights options so only the adapter's
    preprocess_weights (logit_scale fold + untie) runs.
    """
    bridge = _load_cohere_bridge()
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """uses_rms_norm=False must make ln1 mean-subtract (true LayerNorm), not RMS.

        Drives blocks[0].ln1 on a deliberately non-zero-mean input and compares
        against the raw HF CohereLayerNorm (which subtracts the mean). If
        uses_rms_norm regressed to True, the bridge would skip mean subtraction
        and diverge from HF by O(1), tripping this assertion.
        """
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        d_model = cohere_bridge.cfg.d_model
        torch.manual_seed(0)
        # +5.0 shift guarantees a large mean so LN and RMS outputs differ markedly.
        x = torch.randn(1, 4, d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        ln_diff = (bridge_out - hf_out).abs().max().item()
        assert ln_diff < 1e-4, f"ln1 does not match HF CohereLayerNorm: max_diff={ln_diff:.6f}"
        # And it must NOT match an RMS-only (no mean subtraction) normalization.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        rms_diff = (bridge_out - rms_only).abs().max().item()
        assert rms_diff > 1e-2, (
            "ln1 output matches RMS-only normalization; uses_rms_norm must be False "
            f"so the mean is subtracted (rms_diff={rms_diff:.6f})"
        )

    def test_cfg_logit_scale_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: logit_scale must propagate from HF (not silently fall back to 0.0625)."""
        bridge_scale = getattr(cohere_bridge.cfg, "logit_scale")
        assert bridge_scale == cohere_hf.config.logit_scale
        # Anchor 0.125 so a passthrough regression that defaults to 0.0625 also trips here.
        assert bridge_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: rope_parameters must propagate from HF (same passthrough trap as logit_scale)."""
        assert getattr(cohere_bridge.cfg, "rope_parameters") == cohere_hf.config.rope_parameters


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any, cohere_tokens: torch.Tensor
    ) -> None:
        """Compare logits and every block's norm output during real forward passes."""
        reference: dict[str, torch.Tensor] = {}
        norm_modules = {
            f"blocks.{i}.ln1.hook_out": layer.input_layernorm
            for i, layer in enumerate(cohere_hf.model.layers)
        }
        norm_modules["ln_final.hook_out"] = cohere_hf.model.norm
        handles = []
        try:
            for name, module in norm_modules.items():
                handles.append(module.register_forward_hook(_capture_activation(name, reference)))
            with torch.no_grad():
                hf_logits = cohere_hf(cohere_tokens, use_cache=False).logits
                bridge_logits, cache = cohere_bridge.run_with_cache(
                    cohere_tokens, names_filter=list(norm_modules), use_cache=False
                )
        finally:
            for handle in handles:
                handle.remove()

        assert set(reference) == set(norm_modules), "HF did not execute every expected norm"
        for name in norm_modules:
            assert name in cache, f"Bridge did not cache {name}"
            _assert_max_abs_close(
                cache[name],
                reference[name],
                ACTIVATION_ATOL,
                f"{name} ({cohere_tokens.shape[1]} tokens)",
            )
        _assert_max_abs_close(bridge_logits, hf_logits, LOGIT_ATOL, "forward logits")


# ---------------------------------------------------------------------------
# 3. Logit scale applied end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Verify logit_scale is correctly folded into the loaded model.

    process_weights must be called before the fold takes effect — boot_transformers
    alone does NOT call process_weights. cohere_bridge_processed uses a fixture that
    calls process_weights with all standard options disabled so only preprocess_weights
    (the logit_scale fold) runs.
    """

    def test_unembed_weight_is_scaled_relative_to_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After preprocess_weights, lm_head.weight inside the bridge should equal
        # HF lm_head.weight * logit_scale (both [d_vocab, d_model]).
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        tl_weight = cohere_bridge_processed.unembed.original_component.weight  # [d_vocab, d_model]
        hf_weight = cohere_hf.lm_head.weight.detach()  # [d_vocab, d_model]
        expected = hf_weight * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"unembed.weight not correctly scaled: max_diff={max_diff:.6f}, "
            f"logit_scale={logit_scale}"
        )

    def test_unprocessed_logit_scale_not_applied_twice(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Confirm logit_scale isn't double-applied.

        The bridge (before process_weights) delegates to HF's forward, which includes
        the logit_scale multiply. If forward still matches HF, the fold hasn't been
        applied a second time on top of HF's own scale.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        # A second scale multiply changes outputs by the checkpoint-specific logit_scale.
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Possible double-application of logit_scale; diff={max_diff:.6f}"


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Verify preprocess_weights does not corrupt embed.W_E in the tied case.

    Both tests use cohere_bridge_processed (process_weights called with fold-only
    options) because the untie + fold only happens inside process_weights.
    """

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After the fold, embed.W_E must still equal HF's unscaled embed_tokens.weight.
        # If the fold corrupted embed (in-place on the shared tensor), this fails.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (possibly by logit_scale fold): max_diff={max_diff:.6f}"

    @pytest.mark.parametrize("logit_scale", [0.0625, 1.0])
    def test_embed_and_unembed_weights_differ(self, logit_scale: float) -> None:
        # After the logit_scale fold, embed.W_E and unembed.weight must NOT be
        # identical for a non-trivial scale. logit_scale=1.0 is kept as a regression
        # guard for the no-op case, where the two weights stay tied.
        #
        # cfg.logit_scale is set before process_weights so the fold (which reads it
        # inside preprocess_weights) runs with the parametrized value.
        bridge = _load_cohere_bridge()
        setattr(bridge.cfg, "logit_scale", logit_scale)
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        tl_embed = bridge.embed.W_E
        tl_unembed = bridge.unembed.original_component.weight
        weights_identical = torch.allclose(tl_embed, tl_unembed)
        if logit_scale == 1.0:
            assert weights_identical, (
                "embed.W_E and unembed.weight should remain tied when logit_scale=1.0 "
                "(the fold is a no-op)"
            )
        else:
            assert not weights_identical, (
                "embed.W_E and unembed.weight are identical — "
                "logit_scale fold may not have been applied or untied correctly"
            )


# ---------------------------------------------------------------------------
# 5. HF component wiring
# ---------------------------------------------------------------------------


class TestCohereHFComponentWiring:
    """Spot-check that bridge submodules retain the expected HF objects.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache

