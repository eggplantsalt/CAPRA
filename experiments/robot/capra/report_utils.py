"""Report utilities: format and save CAPRA metric summaries.

Phase 1: stub.
Phase 2: implement JSON / CSV / console report writers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from experiments.robot.capra.metrics import AggregateMetrics


def print_aggregate_report(metrics: AggregateMetrics, title: str = "CAPRA Eval") -> None:
    """Pretty-print aggregate metrics to stdout."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  N episodes       : {metrics.n_episodes}")
    print(f"  Success rate     : {metrics.success_rate:.3f}")
    print(f"  SPIR             : {metrics.spir_mean:.4f}")
    print(f"  EAR (J_AR)       : {metrics.ear_mean:.4f}")
    print(f"  EditGain         : {metrics.attribution_edit_gain_mean:.4f}")
    print(f"  LeadTime         : {metrics.precursor_lead_time_mean:.2f} steps")
    print(f"  Prot. displace.  : {metrics.protected_object_displacement_mean:.4f} m")
    print(f"  Topple rate      : {metrics.topple_rate:.3f}")
    print(f"  Support-break    : {metrics.support_break_rate:.3f}")
    print(f"{'='*60}\n")


def save_json_report(
    metrics: AggregateMetrics,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save aggregate metrics as a JSON file."""
    import json
    import dataclasses
    data: Dict[str, Any] = dataclasses.asdict(metrics)
    if extra:
        data.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved CAPRA report to {output_path}")
