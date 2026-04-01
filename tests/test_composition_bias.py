"""Tests for compositional guided sampling.

Tests are split into two groups:
- Parsing tests: pure Python/numpy, no JAX required.
- Sampling integration tests: require JAX + haiku (skipped if unavailable).
"""

import pytest
import numpy as np


# ------------------------------------------------------------------ #
#  Parsing tests (no JAX needed)                                      #
# ------------------------------------------------------------------ #

from crystalformer.src.elements import element_list, element_dict, parse_composition_bias


class TestParseCompositionBias:
    """parse_composition_bias('Fe:2.0,S:1.5') -> float array."""

    def test_single_element(self):
        bias = parse_composition_bias("Fe:2.0")
        assert bias[element_dict["Fe"]] == pytest.approx(2.0)
        assert bias.sum() == pytest.approx(2.0)  # only Fe is nonzero

    def test_multiple_elements(self):
        bias = parse_composition_bias("Fe:2.0,S:1.5,Ni:0.5")
        assert bias[element_dict["Fe"]] == pytest.approx(2.0)
        assert bias[element_dict["S"]] == pytest.approx(1.5)
        assert bias[element_dict["Ni"]] == pytest.approx(0.5)

    def test_negative_bias(self):
        bias = parse_composition_bias("O:-1.0")
        assert bias[element_dict["O"]] == pytest.approx(-1.0)

    def test_empty_string_returns_zeros(self):
        bias = parse_composition_bias("")
        assert np.all(bias == 0)

    def test_none_returns_zeros(self):
        bias = parse_composition_bias(None)
        assert np.all(bias == 0)

    def test_output_shape(self):
        bias = parse_composition_bias("Li:1.0")
        assert bias.shape == (119,)

    def test_custom_atom_types(self):
        bias = parse_composition_bias("H:1.0", atom_types=10)
        assert bias.shape == (10,)
        assert bias[element_dict["H"]] == pytest.approx(1.0)

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Unknown element"):
            parse_composition_bias("Xx:1.0")

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="Invalid bias format"):
            parse_composition_bias("Fe2.0")  # missing colon

    def test_element_outside_atom_types_raises(self):
        """Regression: element index >= atom_types must raise, not silently drop."""
        with pytest.raises(ValueError, match="outside model vocabulary"):
            parse_composition_bias("Fe:2.0", atom_types=10)  # Fe is index 26

    def test_whitespace_tolerance(self):
        bias = parse_composition_bias("  Fe : 2.0 , S : 1.5  ")
        assert bias[element_dict["Fe"]] == pytest.approx(2.0)
        assert bias[element_dict["S"]] == pytest.approx(1.5)

    def test_zero_bias_is_noop(self):
        bias = parse_composition_bias("Fe:0.0")
        assert np.all(bias == 0)

    def test_padding_element_unaffected(self):
        """Index 0 ('X', padding) should stay zero regardless of input."""
        bias = parse_composition_bias("Fe:2.0,S:1.5")
        assert bias[0] == 0.0


# ------------------------------------------------------------------ #
#  Sampling integration tests (need JAX)                              #
# ------------------------------------------------------------------ #

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestBiasAppliedToLogits:
    """Verify that composition_bias shifts atom sampling distribution."""

    def test_positive_bias_increases_probability(self):
        """Element with positive bias should be sampled more often."""
        from crystalformer.src.sample import sample_top_p

        key = jax.random.PRNGKey(42)
        n_samples = 5000
        n_types = 10

        # Uniform logits
        logits = jnp.zeros((n_samples, n_types))

        # No bias -> roughly uniform
        counts_no_bias = np.bincount(
            np.array(sample_top_p(key, logits, 1.0, 1.0)), minlength=n_types
        )

        # Bias toward element 3
        bias = jnp.zeros(n_types).at[3].set(3.0)
        biased_logits = logits + bias
        counts_biased = np.bincount(
            np.array(sample_top_p(key, biased_logits, 1.0, 1.0)),
            minlength=n_types,
        )

        # Element 3 should be sampled much more with bias
        assert counts_biased[3] > counts_no_bias[3] * 1.5

    def test_negative_bias_decreases_probability(self):
        """Element with negative bias should be sampled less often."""
        from crystalformer.src.sample import sample_top_p

        key = jax.random.PRNGKey(42)
        n_samples = 5000
        n_types = 10

        logits = jnp.zeros((n_samples, n_types))

        counts_no_bias = np.bincount(
            np.array(sample_top_p(key, logits, 1.0, 1.0)), minlength=n_types
        )

        bias = jnp.zeros(n_types).at[3].set(-3.0)
        biased_logits = logits + bias
        counts_biased = np.bincount(
            np.array(sample_top_p(key, biased_logits, 1.0, 1.0)),
            minlength=n_types,
        )

        assert counts_biased[3] < counts_no_bias[3] * 0.5

    def test_atom_mask_overrides_positive_bias(self):
        """A masked-out element (atom_mask=False) should stay blocked
        even with large positive bias."""
        from crystalformer.src.sample import sample_top_p

        key = jax.random.PRNGKey(0)
        n_samples = 2000
        n_types = 5

        logits = jnp.zeros((n_samples, n_types))

        # Mask out element 2 (hard block)
        atom_mask = jnp.array([True, True, False, True, True])
        mask_penalty = jnp.where(atom_mask, 0.0, -1e10)

        # Large positive bias on the masked element
        bias = jnp.zeros(n_types).at[2].set(10.0)

        final_logits = logits + mask_penalty + bias
        samples = np.array(sample_top_p(key, final_logits, 1.0, 1.0))

        # Element 2 should never be sampled
        assert np.sum(samples == 2) == 0

    def test_make_sample_crystal_accepts_composition_bias(self):
        """make_sample_crystal should accept composition_bias parameter
        without error (smoke test — does not run sampling)."""
        from crystalformer.src.sample import make_sample_crystal

        import inspect
        sig = inspect.signature(make_sample_crystal)
        assert "composition_bias" in sig.parameters
