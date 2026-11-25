"""Model-agnostic training / evaluation loops.

One :class:`~slz_benchmark.engine.trainer.Trainer` and one
:func:`~slz_benchmark.engine.evaluator.evaluate` serve every model; the model-specific
parts live behind a :class:`~slz_benchmark.adapters.base.SegAdapter`.
"""

from .evaluator import evaluate, format_metrics
from .trainer import Trainer, apply_determinism, poly_scheduler, seed_everything

__all__ = [
    "evaluate",
    "format_metrics",
    "Trainer",
    "apply_determinism",
    "poly_scheduler",
    "seed_everything",
]
