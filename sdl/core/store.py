"""Persistence: experiment store + benchmark recorder.

Deliberately dependency-free (JSON/JSONL on disk) so the framework runs with no
MongoDB/Redis. The interface is small enough to back with a real DB later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sdl.core.experiment import Campaign, ExperimentResult


class ExperimentStore:
    """Append-only JSONL log of results plus a campaign summary file."""

    def __init__(self, root: str | Path, campaign_id: str):
        self.dir = Path(root) / campaign_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.jsonl"
        self.summary_path = self.dir / "campaign.json"

    def append(self, result: ExperimentResult) -> None:
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), default=str) + "\n")

    def save_campaign(self, campaign: Campaign) -> None:
        payload = {
            "summary": campaign.summary(),
            "config": campaign.config,
            "convergence_history": campaign.convergence_history,
            "start_time": campaign.start_time,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load_results(self) -> List[Dict[str, Any]]:
        if not self.results_path.exists():
            return []
        return [json.loads(line) for line in self.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BenchmarkRecorder:
    """Records optimizer-vs-optimizer (and vs baseline) runs for comparison.

    One row per (domain, optimizer, seed) with the convergence trace, so plots
    and statistics can be regenerated without re-running anything.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []

    def add_run(
        self,
        domain: str,
        optimizer: str,
        seed: int,
        convergence: List[float],
        best_scalar: float,
        best_metrics: Optional[Dict[str, float]],
        n_experiments: int,
        wall_time_s: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rows.append(
            {
                "domain": domain,
                "optimizer": optimizer,
                "seed": seed,
                "n_experiments": n_experiments,
                "best_scalar": best_scalar,
                "best_metrics": best_metrics,
                "convergence": convergence,
                "wall_time_s": wall_time_s,
                **(extra or {}),
            }
        )

    def save(self) -> None:
        self.path.write_text(json.dumps(self.rows, indent=2, default=str), encoding="utf-8")

    def summary(self) -> List[Dict[str, Any]]:
        """Aggregate mean/std of final best_scalar per (domain, optimizer)."""
        import statistics
        groups: Dict[tuple, List[float]] = {}
        for r in self.rows:
            groups.setdefault((r["domain"], r["optimizer"]), []).append(r["best_scalar"])
        out = []
        for (domain, opt), vals in sorted(groups.items()):
            out.append(
                {
                    "domain": domain,
                    "optimizer": opt,
                    "runs": len(vals),
                    "mean_best": statistics.fmean(vals),
                    "std_best": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                }
            )
        return out
