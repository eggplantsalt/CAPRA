"""Smoke tests for Phase 7: metrics and eval helpers -- pure Python, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.eval.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    TimestepEvalRecord,
    aggregate_episode_metrics,
    compute_ear,
    compute_episode_metrics,
    compute_spir,
    compute_attribution_edit_gain,
    compute_precursor_lead_time,
)
from experiments.robot.capra.eval.report_utils import (
    print_aggregate_report,
    save_all_reports,
    save_episode_csv,
    save_episode_results,
    save_aggregate_results,
    save_markdown_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(step=0, chosen=0.3, min_eq=0.1, activated=True, dt=0.2,
            topple=0, supp_brk=0, disp=0.05) -> TimestepEvalRecord:
    return TimestepEvalRecord(
        step=step, chosen_footprint=chosen,
        min_equivalent_footprint=min_eq,
        capra_activated=activated, delta_t=dt,
        topple_count=topple, support_break_count=supp_brk,
        protected_object_displacement=disp,
    )


def _episode(n_steps=5, success=True, task_id=0) -> EpisodeMetrics:
    records = [_record(step=i) for i in range(n_steps)]
    return compute_episode_metrics(
        records, episode_id=f"ep_{task_id}",
        task_description="pick up the mug",
        task_id=task_id, success=success,
    )


# ---------------------------------------------------------------------------
# SPIR
# ---------------------------------------------------------------------------

class TestSPIR:
    def test_all_inversions(self):
        assert compute_spir(
            np.array([0.3,0.4,0.5]), np.array([0.1,0.1,0.1]), np.array([True]*3)
        ) == pytest.approx(1.0)

    def test_no_inversions(self):
        assert compute_spir(
            np.array([0.1,0.1,0.1]), np.array([0.1,0.1,0.1]), np.array([True]*3)
        ) == pytest.approx(0.0)

    def test_partial(self):
        assert compute_spir(
            np.array([0.3,0.1,0.4]), np.array([0.1,0.1,0.1]), np.array([True]*3)
        ) == pytest.approx(2/3)

    def test_empty_mask(self):
        assert compute_spir(
            np.array([0.3]), np.array([0.1]), np.array([False])
        ) == pytest.approx(0.0)

    def test_activated_only(self):
        assert compute_spir(
            np.array([0.3,0.3,0.1]), np.array([0.1,0.1,0.1]),
            np.array([True,True,False])
        ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EAR
# ---------------------------------------------------------------------------

class TestEAR:
    def test_basic(self):
        assert compute_ear(
            np.array([0.2,0.4,0.0]), np.array([True]*3)
        ) == pytest.approx(0.2)

    def test_empty_mask(self):
        assert compute_ear(np.array([0.2]), np.array([False])) == pytest.approx(0.0)

    def test_only_activated(self):
        assert compute_ear(
            np.array([0.5,0.0,0.3]), np.array([True,False,True])
        ) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# EditGain / LeadTime
# ---------------------------------------------------------------------------

class TestEditGain:
    def test_full_reduction(self):
        assert compute_attribution_edit_gain(1.0, 0.0) == pytest.approx(1.0)
    def test_no_reduction(self):
        assert compute_attribution_edit_gain(0.5, 0.5) == pytest.approx(0.0, abs=1e-4)
    def test_worse(self):
        assert compute_attribution_edit_gain(0.5, 1.0) < 0.0
    def test_zero_before(self):
        assert np.isfinite(compute_attribution_edit_gain(0.0, 0.0))


class TestLeadTime:
    def test_positive(self):  assert compute_precursor_lead_time(10, 6) == 4
    def test_zero(self):      assert compute_precursor_lead_time(5, 5) == 0
    def test_negative(self):  assert compute_precursor_lead_time(3, 7) < 0


# ---------------------------------------------------------------------------
# TimestepEvalRecord fields
# ---------------------------------------------------------------------------

class TestTimestepRecord:
    def test_fields(self):
        r = _record()
        for f in ["chosen_footprint","min_equivalent_footprint",
                  "capra_activated","delta_t","topple_count",
                  "support_break_count","protected_object_displacement"]:
            assert hasattr(r, f), f"missing field: {f}"

    def test_default_anchor_none(self):
        r = _record()
        assert r.anchor_step is None
        assert r.top_precursor_step is None


# ---------------------------------------------------------------------------
# compute_episode_metrics
# ---------------------------------------------------------------------------

class TestComputeEpisodeMetrics:
    def test_empty(self):
        m = compute_episode_metrics([], success=False)
        assert m.spir == pytest.approx(0.0)
        assert m.ear  == pytest.approx(0.0)
        assert m.total_steps == 0

    def test_fields_populated(self):
        records = [_record(i) for i in range(6)]
        m = compute_episode_metrics(records, success=True,
                                    episode_id="ep1", task_description="t", task_id=2)
        assert m.total_steps == 6
        assert m.n_activated_steps == 6
        assert m.spir > 0.0
        assert m.ear  > 0.0
        assert m.success is True
        assert m.task_id == 2

    def test_no_activated(self):
        records = [_record(activated=False, dt=0.0) for _ in range(4)]
        m = compute_episode_metrics(records, success=False)
        assert m.spir == pytest.approx(0.0)
        assert m.ear  == pytest.approx(0.0)
        assert m.n_activated_steps == 0

    def test_topple_summed(self):
        records = [_record(topple=1), _record(topple=0), _record(topple=2)]
        m = compute_episode_metrics(records, success=False)
        assert m.topple_count == 3

    def test_displacement_summed(self):
        records = [_record(disp=0.1), _record(disp=0.2)]
        m = compute_episode_metrics(records, success=False)
        assert m.protected_object_displacement == pytest.approx(0.3)

    def test_raw_arrays_stored(self):
        records = [_record(i) for i in range(3)]
        m = compute_episode_metrics(records, success=False)
        assert len(m.chosen_footprints) == 3
        assert len(m.activated_mask) == 3
        assert len(m.delta_t_values) == 3


# ---------------------------------------------------------------------------
# aggregate_episode_metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_empty(self):
        agg = aggregate_episode_metrics([])
        assert agg.n_episodes == 0
        assert agg.success_rate == pytest.approx(0.0)

    def test_success_rate(self):
        eps = [_episode(success=True), _episode(success=False), _episode(success=True)]
        agg = aggregate_episode_metrics(eps)
        assert agg.success_rate == pytest.approx(2/3)

    def test_spir_range(self):
        eps = [_episode() for _ in range(4)]
        agg = aggregate_episode_metrics(eps)
        assert 0.0 <= agg.spir_mean <= 1.0

    def test_std_nonneg(self):
        eps = [_episode() for _ in range(4)]
        agg = aggregate_episode_metrics(eps)
        assert agg.spir_std >= 0.0
        assert agg.ear_std  >= 0.0

    def test_n_episodes(self):
        eps = [_episode() for _ in range(7)]
        agg = aggregate_episode_metrics(eps, n_tasks=3)
        assert agg.n_episodes == 7
        assert agg.n_tasks == 3

    def test_required_fields(self):
        eps = [_episode()]
        agg = aggregate_episode_metrics(eps)
        for f in ["success_rate","spir_mean","spir_std","ear_mean","ear_std",
                  "attribution_edit_gain_mean","precursor_lead_time_mean",
                  "protected_object_displacement_mean",
                  "topple_rate","support_break_rate","activation_rate_mean"]:
            assert hasattr(agg, f), f"missing field: {f}"


# ---------------------------------------------------------------------------
# report_utils
# ---------------------------------------------------------------------------

class TestReportUtils:
    def _data(self):
        eps = [_episode(n_steps=5, success=(i % 2 == 0), task_id=i % 3)
               for i in range(6)]
        return eps, aggregate_episode_metrics(eps, n_tasks=3)

    def test_print_console(self, capsys):
        _, agg = self._data()
        print_aggregate_report(agg, title="Test")
        out = capsys.readouterr().out
        assert "SPIR" in out and "EAR" in out

    def test_episode_json(self, tmp_path):
        eps, _ = self._data()
        p = tmp_path / "ep.json"
        save_episode_results(eps, p)
        import json
        data = json.load(open(p))
        assert len(data) == 6
        for k in ["spir","ear","success","topple_count"]: assert k in data[0]

    def test_aggregate_json(self, tmp_path):
        eps, agg = self._data()
        p = tmp_path / "agg.json"
        save_aggregate_results(agg, p, extra={"model_path": "test"})
        import json
        data = json.load(open(p))
        assert "spir_mean" in data and "model_path" in data

    def test_csv(self, tmp_path):
        eps, _ = self._data()
        p = tmp_path / "ep.csv"
        save_episode_csv(eps, p)
        import csv
        rows = list(csv.DictReader(open(p)))
        assert len(rows) == 6 and "spir" in rows[0]

    def test_markdown(self, tmp_path):
        eps, agg = self._data()
        p = tmp_path / "s.md"
        save_markdown_summary(agg, eps, p,
                              model_path="test_model", task_suite="libero_spatial")
        text = p.read_text()
        assert "SPIR" in text and "EAR" in text and "Success" in text

    def test_save_all(self, tmp_path):
        eps, agg = self._data()
        save_all_reports(agg, eps, tmp_path,
                         model_path="m", task_suite="libero_spatial")
        for fn in ["results_episodes.json","results_aggregate.json",
                   "results_episodes.csv","summary.md"]:
            assert (tmp_path / fn).exists(), f"missing: {fn}"


# ---------------------------------------------------------------------------
# Baseline mode: null CAPRA values, external metrics still collected
# ---------------------------------------------------------------------------

class TestBaselineMode:
    def test_spir_ear_zero_no_counterfactuals(self):
        records = [
            TimestepEvalRecord(step=i, chosen_footprint=0.3,
                               min_equivalent_footprint=0.3,
                               capra_activated=False, delta_t=0.0)
            for i in range(10)
        ]
        m = compute_episode_metrics(records, success=True)
        assert m.spir == pytest.approx(0.0)
        assert m.ear  == pytest.approx(0.0)
        assert m.n_activated_steps == 0

    def test_external_metrics_collected(self):
        records = [
            TimestepEvalRecord(
                step=i, chosen_footprint=0.1, min_equivalent_footprint=0.1,
                capra_activated=False, delta_t=0.0,
                topple_count=1 if i == 3 else 0,
                protected_object_displacement=0.05,
            )
            for i in range(5)
        ]
        m = compute_episode_metrics(records, success=False)
        assert m.topple_count > 0
        assert m.protected_object_displacement > 0.0

