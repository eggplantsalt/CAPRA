"""Tests for precursor.precursor_loss_weight."""
import pytest

from experiments.robot.capra.precursor import precursor_loss_weight, PrecursorChain, PrecursorEntry
import numpy as np


def test_weight_formula():
    w = precursor_loss_weight(delta_t=0.2, r_t=0.5, rho=1.0)
    assert w == pytest.approx(0.2 * (1.0 + 1.0 * 0.5))


def test_weight_zero_delta():
    w = precursor_loss_weight(delta_t=0.0, r_t=0.9, rho=2.0)
    assert w == pytest.approx(0.0)


def test_chain_get_weight_missing_step():
    chain = PrecursorChain(anchor_step=10, window=5, entries=[])
    assert chain.get_weight(3) == pytest.approx(0.0)


def test_chain_top_k():
    entries = [
        PrecursorEntry(step=1, delta_hazard=0.1, attribution_score=0.3,
                       replacement_action=np.zeros((8, 7))),
        PrecursorEntry(step=2, delta_hazard=0.4, attribution_score=0.9,
                       replacement_action=np.zeros((8, 7))),
        PrecursorEntry(step=3, delta_hazard=0.2, attribution_score=0.5,
                       replacement_action=np.zeros((8, 7))),
    ]
    chain = PrecursorChain(anchor_step=5, window=5, entries=entries)
    top1 = chain.top_k(1)
    assert top1[0].step == 2
