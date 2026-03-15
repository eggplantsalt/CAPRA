# ===== CAPRA 结果报告生成器 (report_utils.py) =====
# 主入口：save_all_reports(aggregate, episodes, output_dir, model_path, task_suite)
#
# 输出四个文件：
#   results_aggregate.json   聚合指标 + 运行元数据
#   results_episodes.json    每个 episode 的详细指标
#   results_episodes.csv     表格格式，方便 pandas/Excel
#   summary.md               人类可读 Markdown（含每任务分组表格）

"""Report utilities: format and save CAPRA metric summaries.

Outputs
-------
  <run_dir>/results_episodes.json   -- per-episode metrics (machine-readable)
  <run_dir>/results_aggregate.json  -- aggregate metrics   (machine-readable)
  <run_dir>/results_episodes.csv    -- per-episode metrics (tabular)
  <run_dir>/summary.md              -- human-readable markdown summary

Schema
------
results_episodes.json:  list of dicts, one per episode.
  Keys: episode_id, task_description, task_id, success,
        total_steps, n_activated_steps, spir, ear,
        attribution_edit_gain, precursor_lead_time,
        protected_object_displacement, topple_count, support_break_count

results_aggregate.json: single dict.
  Keys: n_episodes, n_tasks, success_rate,
        spir_mean, spir_std, ear_mean, ear_std,
        attribution_edit_gain_mean, precursor_lead_time_mean,
        protected_object_displacement_mean,
        topple_rate, support_break_rate, activation_rate_mean,
        + run metadata (model_path, task_suite, timestamp, seed)
"""
from __future__ import annotations

import csv
import dataclasses
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.robot.capra.metrics import AggregateMetrics, EpisodeMetrics


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def print_aggregate_report(
    metrics: AggregateMetrics,
    title: str = "CAPRA Eval",
    model_path: str = "",
    task_suite: str = "",
) -> None:
    """Pretty-print aggregate metrics to stdout."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    if model_path:
        print(f"  Model : {model_path}")
    if task_suite:
        print(f"  Suite : {task_suite}")
    print(sep)
    print(f"  Episodes          : {metrics.n_episodes}")
    print(f"  Tasks             : {metrics.n_tasks}")
    print(f"  Success rate      : {metrics.success_rate:.3f}")
    print("  --- Primary ---")
    print(f"  SPIR              : {metrics.spir_mean:.4f}  (std {metrics.spir_std:.4f})")
    print(f"  EAR (J_AR)        : {metrics.ear_mean:.4f}  (std {metrics.ear_std:.4f})")
    print("  --- Mechanism ---")
    print(f"  EditGain          : {metrics.attribution_edit_gain_mean:.4f}")
    print(f"  LeadTime          : {metrics.precursor_lead_time_mean:.2f} steps")
    print("  --- External ---")
    print(f"  Prot. displace.   : {metrics.protected_object_displacement_mean:.4f} m")
    print(f"  Topple rate       : {metrics.topple_rate:.3f}")
    print(f"  Support-break rate: {metrics.support_break_rate:.3f}")
    print("  --- Activation ---")
    print(f"  Activation rate   : {metrics.activation_rate_mean:.3f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# JSON writers
# ---------------------------------------------------------------------------

def _episode_to_dict(ep: EpisodeMetrics) -> Dict[str, Any]:
    """Serialize one EpisodeMetrics to a JSON-safe dict (no raw arrays)."""
    return {
        "episode_id":                    ep.episode_id,
        "task_description":              ep.task_description,
        "task_id":                       ep.task_id,
        "success":                       ep.success,
        "total_steps":                   ep.total_steps,
        "n_activated_steps":             ep.n_activated_steps,
        "spir":                          round(ep.spir, 6),
        "ear":                           round(ep.ear, 6),
        "attribution_edit_gain":         round(ep.attribution_edit_gain, 6),
        "precursor_lead_time":           round(ep.precursor_lead_time, 2),
        "protected_object_displacement": round(ep.protected_object_displacement, 6),
        "topple_count":                  ep.topple_count,
        "support_break_count":           ep.support_break_count,
    }


def save_episode_results(
    episodes: List[EpisodeMetrics],
    output_path: Path,
) -> None:
    """Save per-episode results as a JSON list."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [_episode_to_dict(ep) for ep in episodes]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[report] Episode results -> {output_path}  ({len(data)} episodes)")


def save_aggregate_results(
    metrics: AggregateMetrics,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save aggregate metrics as a JSON dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = dataclasses.asdict(metrics)
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    if extra:
        data.update(extra)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[report] Aggregate results -> {output_path}")


# Convenience alias kept for backwards compat
def save_json_report(
    metrics: AggregateMetrics,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    save_aggregate_results(metrics, output_path, extra)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_EPISODE_CSV_FIELDS = [
    "episode_id", "task_description", "task_id", "success",
    "total_steps", "n_activated_steps",
    "spir", "ear",
    "attribution_edit_gain", "precursor_lead_time",
    "protected_object_displacement", "topple_count", "support_break_count",
]


def save_episode_csv(
    episodes: List[EpisodeMetrics],
    output_path: Path,
) -> None:
    """Save per-episode results as a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_episode_to_dict(ep) for ep in episodes]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] Episode CSV     -> {output_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------

def save_markdown_summary(
    aggregate: AggregateMetrics,
    episodes: List[EpisodeMetrics],
    output_path: Path,
    model_path: str = "",
    task_suite: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a short markdown summary file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CAPRA Eval Summary",
        "",
        f"**Generated**: {ts}  ",
        f"**Model**: `{model_path or 'unknown'}`  ",
        f"**Suite**: `{task_suite or 'unknown'}`  ",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Episodes | {aggregate.n_episodes} |",
        f"| Tasks | {aggregate.n_tasks} |",
        f"| **Success rate** | **{aggregate.success_rate:.3f}** |",
        f"| **SPIR** | **{aggregate.spir_mean:.4f}** \u00b1 {aggregate.spir_std:.4f} |",
        f"| **EAR (J_AR)** | **{aggregate.ear_mean:.4f}** \u00b1 {aggregate.ear_std:.4f} |",
        f"| EditGain | {aggregate.attribution_edit_gain_mean:.4f} |",
        f"| LeadTime (steps) | {aggregate.precursor_lead_time_mean:.2f} |",
        f"| Prot. displacement (m) | {aggregate.protected_object_displacement_mean:.4f} |",
        f"| Topple rate | {aggregate.topple_rate:.3f} |",
        f"| Support-break rate | {aggregate.support_break_rate:.3f} |",
        f"| Activation rate | {aggregate.activation_rate_mean:.3f} |",
        "",
    ]

    if extra:
        lines += ["## Run Info", ""]
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Per-task breakdown (group episodes by task_id)
    if episodes:
        by_task: Dict[int, List[EpisodeMetrics]] = defaultdict(list)
        for ep in episodes:
            by_task[ep.task_id].append(ep)

        lines += ["## Per-Task Results", ""]
        lines += [
            "| task_id | task | n_ep | succ | SPIR | EAR | topple | supp_brk |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for tid in sorted(by_task.keys()):
            eps = by_task[tid]
            desc = eps[0].task_description[:40] + ("..." if len(eps[0].task_description) > 40 else "")
            sr = float(np.mean([float(e.success) for e in eps])) if eps else 0.0
            sp = float(np.mean([e.spir for e in eps])) if eps else 0.0
            ea = float(np.mean([e.ear for e in eps])) if eps else 0.0
            tp = float(np.mean([float(e.topple_count > 0) for e in eps])) if eps else 0.0
            sb = float(np.mean([float(e.support_break_count > 0) for e in eps])) if eps else 0.0
            lines.append(
                f"| {tid} | {desc} | {len(eps)} "
                f"| {sr:.2f} | {sp:.3f} | {ea:.3f} | {tp:.2f} | {sb:.2f} |"
            )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Markdown summary -> {output_path}")


# ---------------------------------------------------------------------------
# Save all outputs at once
# ---------------------------------------------------------------------------

def save_all_reports(
    aggregate: AggregateMetrics,
    episodes: List[EpisodeMetrics],
    run_dir: Path,
    model_path: str = "",
    task_suite: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write all four output files under run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"model_path": model_path, "task_suite": task_suite}
    if extra:
        meta.update(extra)
    save_episode_results(episodes,    run_dir / "results_episodes.json")
    save_aggregate_results(aggregate, run_dir / "results_aggregate.json", meta)
    save_episode_csv(episodes,        run_dir / "results_episodes.csv")
    save_markdown_summary(
        aggregate, episodes, run_dir / "summary.md",
        model_path=model_path, task_suite=task_suite, extra=extra,
    )
