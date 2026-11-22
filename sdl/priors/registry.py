"""Curated priors library — the system of record for the optimizer's head-starts.

WHAT A "PRIOR" IS (plain words)
    A *prior* is a cheap predictor of an experiment's outcome that the optimizer
    starts from, so it doesn't learn from a blank slate. In code, a prior is just
    a domain's ``simulate()`` method (see sdl/core/domain.py) — a textbook formula
    (mechanistic) or a model fit on past data (data-driven). ``physics_bo`` reads
    it via ``_prior_scalar`` (see sdl/optimizers/physics_bo.py) and only learns
    the *gap* between the prior and reality.

WHAT THIS MODULE IS
    Bookkeeping around those priors so the library can be maintained and
    improved over time: every prior gets a "card" recording where it
    came from, how well it matches reality, where it's known to be wrong, who owns
    it, and its version history. Cards are plain JSON files in sdl/priors/cards/.

WHAT THIS MODULE IS NOT
    It does not change how optimizers run. It is a curator's notebook, not part of
    the optimization loop. A prior with no card still works; the card is how a team
    tracks quality and improves priors over time.

See sdl/priors/cards/TEMPLATE.json for the card format.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

_CARDS_DIR = os.path.join(os.path.dirname(__file__), "cards")


def cards_dir() -> str:
    return _CARDS_DIR


@dataclass
class PriorCard:
    """Metadata + validation record for one prior. One JSON file per card.

    Required identity fields are first; everything else has a sensible default so
    a scientist can fill the card incrementally as evidence accumulates.
    """

    prior_id: str                 # unique, e.g. "buchwald_hartwig.v2"
    reaction_class: str           # human label, e.g. "Buchwald-Hartwig amination"
    domain: str                   # registry domain it backs, e.g. "reaction" / "buchwald"
    kind: str = "mechanistic"     # "mechanistic" | "data_driven" | "hybrid"
    version: int = 1
    source: str = ""              # papers / datasets / lab campaigns it came from
    description: str = ""         # what it predicts and its key assumptions

    # --- validation: how good is this prior? (fill via score_prior) ---
    objective: str = ""           # which metric was scored, e.g. "yield"
    n_validation: int = 0         # how many measured points it was checked against
    rmse: Optional[float] = None  # root-mean-square error vs measured (objective units)
    bias: Optional[float] = None  # mean(prior - measured); +ve = prior over-predicts
    spearman: Optional[float] = None  # rank correlation prior-vs-measured (0..1 good)
    coverage: Optional[float] = None  # fraction of the design space the data covers
    reality_gap_notes: str = ""   # where the prior is known to be wrong (the residual)

    # --- governance ---
    owner: str = ""               # who maintains it
    status: str = "draft"         # "draft" | "validated" | "deprecated"
    created_utc: float = field(default_factory=lambda: time.time())
    updated_utc: float = field(default_factory=lambda: time.time())
    history: List[str] = field(default_factory=list)  # human-readable change log

    def touch(self, note: str) -> None:
        """Record a change in the card's history and bump the timestamp."""
        stamp = time.strftime("%Y-%m-%d", time.gmtime())
        self.history.append(f"{stamp}: {note}")
        self.updated_utc = time.time()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PriorCard":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _card_path(prior_id: str) -> str:
    safe = prior_id.replace("/", "_")
    return os.path.join(_CARDS_DIR, f"{safe}.json")


def save_card(card: PriorCard) -> str:
    os.makedirs(_CARDS_DIR, exist_ok=True)
    path = _card_path(card.prior_id)
    with open(path, "w") as fh:
        json.dump(card.to_dict(), fh, indent=2, sort_keys=True)
    return path


def load_card(prior_id: str) -> PriorCard:
    with open(_card_path(prior_id)) as fh:
        return PriorCard.from_dict(json.load(fh))


def list_cards() -> List[PriorCard]:
    if not os.path.isdir(_CARDS_DIR):
        return []
    out: List[PriorCard] = []
    for fn in sorted(os.listdir(_CARDS_DIR)):
        if fn.endswith(".json") and not fn.startswith("TEMPLATE"):
            try:
                with open(os.path.join(_CARDS_DIR, fn)) as fh:
                    out.append(PriorCard.from_dict(json.load(fh)))
            except Exception:
                pass
    return out


def score_prior(predicted: Sequence[float], measured: Sequence[float]) -> Dict[str, float]:
    """Quantify how well a prior matches reality. Feed it the prior's predictions
    and the real measured values for the same points; returns rmse, bias, and a
    rank correlation (spearman) you can drop straight into a PriorCard.

    This is the number that justifies trusting (or retiring) a prior, and the
    quantity that should *shrink* as more data improves the prior.
    """
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    if p.shape != m.shape or p.size == 0:
        raise ValueError("predicted and measured must be same-length, non-empty")
    rmse = float(np.sqrt(np.mean((p - m) ** 2)))
    bias = float(np.mean(p - m))
    spearman = _spearman(p, m)
    return {"n_validation": int(p.size), "rmse": rmse, "bias": bias, "spearman": spearman}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / denom) if denom else float("nan")
