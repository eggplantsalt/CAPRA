"""Tests for precursor attribution -- pure numpy, no env required.

Covers:
1. precursor_loss_weight formula
2. PrecursorChain.get_weight / top_k
3. measure_downstream_hazard
4. compute_precursor_chain_from_footprints:
   - toy chain case: true precursor step reduces hazard more than random
   - lead-time case: hazard rises several steps before terminal event
5. AttributionEditGain (metrics.py)
6. PrecursorLeadTime (metrics.py)
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.capra_config import CAPRAConfig
from experiments.robot.capra.precursor import (
    PrecursorEntry,
    PrecursorChain,
    precursor_loss_weight,
    measure_downstream_hazard,
    compute_precursor_chain_from_footprints,
)
from experiments.robot.capra.metrics import (
    compute_attribution_edit_gain,
    compute_precursor_lead_time,
)


# ---------------------------------------------------------------------------
# precursor_loss_weight
# ---------------------------------------------------------------------------

class TestLossWeight:
    def test_formula(self):
        w = precursor_loss_weight(delta_t=0.2, r_t=0.5, rho=1.0)
        assert w == pytest.approx(0.2 * 1.5)

    def test_zero_delta_gives_zero_weight(self):
        w = precursor_loss_weight(delta_t=0.0, r_t=0.9, rho=2.0)
        assert w == pytest.approx(0.0)

    def test_zero_r_t_returns_delta_t(self):
        w = precursor_loss_weight(delta_t=0.3, r_t=0.0, rho=5.0)
        assert w == pytest.approx(0.3)

    def test_rho_upweights(self):
        w_low  = precursor_loss_weight(delta_t=0.2, r_t=0.5, rho=0.0)
        w_high = precursor_loss_weight(delta_t=0.2, r_t=0.5, rho=2.0)
        assert w_high > w_low


# ---------------------------------------------------------------------------
# PrecursorChain helpers
# ---------------------------------------------------------------------------

class TestPrecursorChain:
    def _chain(self):
        entries = [
            PrecursorEntry(step=1, delta_hazard=0.1, attribution_score=0.15,
                           replacement_action=np.zeros((8, 7))),
            PrecursorEntry(step=2, delta_hazard=0.4, attribution_score=0.60,
                           replacement_action=np.zeros((8, 7))),
            PrecursorEntry(step=3, delta_hazard=0.2, attribution_score=0.25,
                           replacement_action=np.zeros((8, 7))),
        ]
        return PrecursorChain(anchor_step=5, window=5, entries=entries)

    def test_get_weight_present(self):
        chain = self._chain()
        assert chain.get_weight(2) == pytest.approx(0.60)

    def test_get_weight_missing(self):
        chain = self._chain()
        assert chain.get_weight(99) == pytest.approx(0.0)

    def test_top_k_order(self):
        chain = self._chain()
        top1 = chain.top_k(1)
        assert top1[0].step == 2

    def test_top_k_returns_k(self):
        chain = self._chain()
        assert len(chain.top_k(2)) == 2
        assert len(chain.top_k(10)) == 3   # capped at chain size

    def test_is_empty_false(self):
        assert not self._chain().is_empty()

    def test_is_empty_true(self):
        assert PrecursorChain(anchor_step=5, window=5).is_empty()


# ---------------------------------------------------------------------------
# measure_downstream_hazard
# ---------------------------------------------------------------------------

class TestMeasureDownstreamHazard:
    def test_full_sequence(self):
        fp = np.array([0.1, 0.2, 0.3, 0.4])
        assert measure_downstream_hazard(fp, 0) == pytest.approx(1.0)

    def test_partial_sequence(self):
        fp = np.array([0.1, 0.2, 0.3, 0.4])
        assert measure_downstream_hazard(fp, 2) == pytest.approx(0.7)

    def test_past_end_returns_zero(self):
        fp = np.array([0.1, 0.2])
        assert measure_downstream_hazard(fp, 5) == pytest.approx(0.0)

    def test_single_step(self):
        fp = np.array([0.5])
        assert measure_downstream_hazard(fp, 0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TOY CHAIN CASE
# ---------------------------------------------------------------------------

class TestToyChainCase:
    """Prove that the true precursor step gets higher attribution than a
    random / low-impact step.

    Scenario
    --------
    Window W = 5 steps; anchor at step 10.
    Step 7 (window index 2) is the TRUE precursor:
      original F = 0.8  (high risk action)
      replacement F = 0.1  (safe alternative drops hazard by 0.7)

    Step 6 (window index 1) is a RANDOM / low-impact step:
      original F = 0.3
      replacement F = 0.25  (barely changes anything)

    Expected: attribution_score[step=7] > attribution_score[step=6]
    """
    def _run(self):
        cfg = CAPRAConfig(attribution_max_steps=5)
        window_start = 6   # absolute step of first window element
        anchor_step  = 10
        W = 5

        # Original footprints: [0.1, 0.3, 0.8, 0.2, 0.1]  (indices 0..4)
        step_fp = np.array([0.1, 0.3, 0.8, 0.2, 0.1], dtype=np.float32)

        # Replacement footprints (hypothetical F after best alternative)
        # Index 2 (step 8): replacement drops 0.8 -> 0.1  (delta = 0.7)
        # Index 1 (step 7): replacement drops 0.3 -> 0.25 (delta = 0.05)
        # Others unchanged
        repl_fp = np.array([0.1, 0.25, 0.1, 0.2, 0.1], dtype=np.float32)

        chain = compute_precursor_chain_from_footprints(
            step_footprints=step_fp,
            replacement_footprints=repl_fp,
            anchor_step=anchor_step,
            window_start_step=window_start,
            cfg=cfg,
        )
        return chain

    def test_chain_nonempty(self):
        assert not self._run().is_empty()

    def test_true_precursor_highest_score(self):
        """True precursor (window index 2, step 8) must have highest R_t."""
        chain = self._run()
        top = chain.top_k(1)[0]
        assert top.step == 8  # window_start(6) + index(2) = 8

    def test_true_precursor_score_gt_random(self):
        """True precursor attribution score exceeds random step."""
        chain = self._run()
        score_true   = chain.get_weight(8)  # true precursor
        score_random = chain.get_weight(7)  # low-impact step
        assert score_true > score_random

    def test_scores_sum_to_one(self):
        chain = self._run()
        total = sum(e.attribution_score for e in chain.entries)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_delta_hazard_positive(self):
        chain = self._run()
        for e in chain.entries:
            assert e.delta_hazard >= 0.0

    def test_entries_time_ordered(self):
        chain = self._run()
        steps = [e.step for e in chain.entries]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# LEAD-TIME CASE
# ---------------------------------------------------------------------------

class TestLeadTimeCase:
    """Prove that precursor attribution identifies steps BEFORE the terminal
    hazard -- i.e. meaningful lead-time exists.

    Scenario
    --------
    Window W = 8; anchor at step 20.
    Steps 13..20 (window indices 0..7).

    The true precursor is at step 14 (window index 1) -- 6 steps before anchor.
    Terminal hazard step is 20 (window index 7).

    We verify:
      1. top precursor step < anchor step  (positive lead time)
      2. lead_time == anchor_step - top_precursor_step >= 4
    """
    def _chain(self):
        cfg = CAPRAConfig(attribution_max_steps=8)
        # Original F: low until step 14 (index 1) which has high F
        step_fp = np.array([0.05, 0.90, 0.10, 0.10, 0.15, 0.10, 0.10, 0.80],
                           dtype=np.float32)
        # Replacement: index 1 drops from 0.90 to 0.05 (true early precursor)
        repl_fp = np.array([0.05, 0.05, 0.10, 0.10, 0.15, 0.10, 0.10, 0.75],
                           dtype=np.float32)
        return compute_precursor_chain_from_footprints(
            step_footprints=step_fp,
            replacement_footprints=repl_fp,
            anchor_step=20,
            window_start_step=13,
            cfg=cfg,
        )

    def test_top_precursor_before_anchor(self):
        chain = self._chain()
        top = chain.top_k(1)[0]
        assert top.step < 20   # precursor is strictly before anchor

    def test_lead_time_positive(self):
        chain = self._chain()
        top = chain.top_k(1)[0]
        lead_time = compute_precursor_lead_time(
            anchor_step=20,
            top_precursor_step=top.step,
        )
        assert lead_time > 0

    def test_lead_time_at_least_4_steps(self):
        """True precursor (step 14) is 6 steps before anchor (step 20)."""
        chain = self._chain()
        top = chain.top_k(1)[0]
        assert top.step == 14   # window_start(13) + index(1)
        lead_time = compute_precursor_lead_time(20, top.step)
        assert lead_time >= 4

    def test_attribution_rises_before_hazard(self):
        """The step before the terminal event has non-zero attribution."""
        chain = self._chain()
        # At least one entry should have step <= anchor_step - 4
        early_entries = [e for e in chain.entries if e.step <= 16]
        assert len(early_entries) > 0


# ---------------------------------------------------------------------------
# AttributionEditGain
# ---------------------------------------------------------------------------

class TestAttributionEditGain:
    def test_positive_gain(self):
        gain = compute_attribution_edit_gain(hazard_before=1.0, hazard_after=0.3)
        assert gain == pytest.approx(0.7 / 1.0000001, rel=1e-4)

    def test_zero_gain_no_change(self):
        gain = compute_attribution_edit_gain(hazard_before=0.5, hazard_after=0.5)
        assert gain == pytest.approx(0.0, abs=1e-5)

    def test_negative_gain_worse(self):
        gain = compute_attribution_edit_gain(hazard_before=0.3, hazard_after=0.5)
        assert gain < 0.0

    def test_zero_before_near_zero_denominator(self):
        gain = compute_attribution_edit_gain(hazard_before=0.0, hazard_after=0.0)
        assert gain == pytest.approx(0.0, abs=1e-4)

    def test_full_hazard_elimination(self):
        gain = compute_attribution_edit_gain(hazard_before=1.0, hazard_after=0.0)
        assert gain > 0.99


# ---------------------------------------------------------------------------
# PrecursorLeadTime
# ---------------------------------------------------------------------------

class TestPrecursorLeadTime:
    def test_basic_lead_time(self):
        lt = compute_precursor_lead_time(anchor_step=20, top_precursor_step=14)
        assert lt == 6

    def test_zero_lead_time(self):
        lt = compute_precursor_lead_time(anchor_step=10, top_precursor_step=10)
        assert lt == 0

    def test_one_step_lead_time(self):
        lt = compute_precursor_lead_time(anchor_step=5, top_precursor_step=4)
        assert lt == 1

    def test_negative_is_data_error(self):
        """If precursor is after anchor, result is negative (data error indicator)."""
        lt = compute_precursor_lead_time(anchor_step=5, top_precursor_step=7)
        assert lt < 0


# ---------------------------------------------------------------------------
# Integration: chain -> edit_gain -> lead_time pipeline
# ---------------------------------------------------------------------------

class TestFullAttributionPipeline:
    """End-to-end: build chain, compute edit gain and lead time from it."""

    def test_pipeline_toy_chain(self):
        cfg = CAPRAConfig(attribution_max_steps=5)
        anchor = 10
        window_start = 6

        step_fp = np.array([0.1, 0.3, 0.8, 0.2, 0.1], dtype=np.float32)
        repl_fp = np.array([0.1, 0.25, 0.1, 0.2, 0.1], dtype=np.float32)

        chain = compute_precursor_chain_from_footprints(
            step_fp, repl_fp, anchor_step=anchor,
            window_start_step=window_start, cfg=cfg
        )
        assert not chain.is_empty()

        top = chain.top_k(1)[0]

        # Edit gain
        orig_fp_of_top = float(step_fp[top.step - window_start])
        repl_fp_of_top = float(repl_fp[top.step - window_start])
        gain = compute_attribution_edit_gain(orig_fp_of_top, repl_fp_of_top)
        assert gain > 0

        # Lead time
        lt = compute_precursor_lead_time(anchor, top.step)
        assert lt >= 0

    def test_loss_weight_with_precursor(self):
        """w_t is strictly larger when R_t > 0 vs R_t = 0."""
        delta_t = 0.3
        rho     = 0.5
        w_no_attr = precursor_loss_weight(delta_t, r_t=0.0, rho=rho)
        w_with    = precursor_loss_weight(delta_t, r_t=0.8, rho=rho)
        assert w_with > w_no_attr
