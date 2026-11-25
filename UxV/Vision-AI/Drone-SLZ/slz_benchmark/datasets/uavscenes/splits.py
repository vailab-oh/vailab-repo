"""Split scheme -> manifest path, shared by training and evaluation.

Schemes:
  - ``loso``          : ``<root>/manifests/loso/fold<N>/<split>.csv`` (N in 1..4). The
    paper's split: leave-one-sequence-out, with a 30-frame temporal guard and a 5 m
    spatial guard between train and val. Installed by ``prepare_data.py``.
  - ``loso_temporal`` : ``<root>/manifests/loso_temporal/fold<N>/<split>.csv``. The same
    split with the temporal guard only; an intermediate step when the splits are rebuilt
    (``tools/make_loso_splits.py`` -> ``tools/apply_spatial_guard.py``).
  - ``fixed`` / ``four_fold``: the family splits the label generator writes
    (``manifests/fixed/<split>.csv``, ``manifests/four_fold/fold_<N>/<split>.csv``);
    ``make_loso_splits.py`` reads them.

Fold directories may be spelled ``fold<N>`` or ``fold_<N>``; ``manifest_path`` returns
the spelling that exists on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

VALID_FOLDS = (1, 2, 3, 4)

# scheme -> (subdir, fold-dir spellings, needs a fold)
_SCHEMES = {
    "fixed": ("fixed", (), False),
    "four_fold": ("four_fold", ("fold_{n}", "fold{n}"), True),
    "loso_temporal": ("loso_temporal", ("fold{n}", "fold_{n}"), True),
    "loso": ("loso", ("fold{n}", "fold_{n}"), True),
}

SCHEMES = tuple(_SCHEMES)


def canonical_scheme(scheme: str) -> str:
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown split scheme {scheme!r}; expected one of {SCHEMES}.")
    return scheme


def scheme_needs_fold(scheme: str) -> bool:
    return _SCHEMES[canonical_scheme(scheme)][2]


def validate_fold(scheme: str, fold: Optional[int]) -> None:
    """Fail fast for a missing or invalid fold."""
    canon = canonical_scheme(scheme)
    if _SCHEMES[canon][2]:
        if fold is None:
            raise ValueError(f"--fold is required for scheme {canon!r}")
        if int(fold) not in VALID_FOLDS:
            raise ValueError(
                f"--fold must be one of {VALID_FOLDS} for scheme {canon!r}, got {fold}"
            )
    elif fold is not None:
        raise ValueError(f"scheme {canon!r} does not take a --fold (got {fold})")


def _check_spatial_guard(base: Path) -> None:
    """``manifests/loso`` must hold the spatially guarded split. A ``split_info.json``
    without ``d_guard_m`` means temporal-only manifests are sitting there."""
    info = base / "split_info.json"
    if info.exists() and "d_guard_m" not in json.loads(info.read_text()):
        raise ValueError(
            f"{base} holds temporal-only LOSO manifests (no d_guard_m in split_info.json). "
            "Install the paper's splits with: python prepare_data.py --stage splits --root <PREPARED>"
        )


def manifest_path(
    root: str | Path,
    split: str,
    scheme: str = "loso",
    fold: Optional[int] = None,
) -> Path:
    root = Path(root)
    canon = canonical_scheme(scheme)
    subdir, fold_specs, needs_fold = _SCHEMES[canon]
    validate_fold(canon, fold)

    base = root / "manifests" / subdir
    if canon == "loso":
        _check_spatial_guard(base)
    if not needs_fold:
        return base / f"{split}.csv"

    n = int(fold)
    candidates = [base / spec.format(n=n) / f"{split}.csv" for spec in fold_specs]
    for cand in candidates:
        if cand.exists():
            return cand
    # Nothing on disk yet: return the canonical spelling.
    return candidates[0]
