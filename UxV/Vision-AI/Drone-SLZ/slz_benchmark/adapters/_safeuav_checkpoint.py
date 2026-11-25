"""Loader for the official SafeUAV pretrained checkpoints (``HVO_small.pkl``,
``HVO_large.pkl``).

The files are a ``torch.save`` of a dict, written by SafeUAV's ``neural-wrappers``
dependency, whose ``"weights"`` entry is a *positional* list of parameter tensors
(``model.parameters()`` order, no names). The same dict also holds optimizer / scheduler /
callback state built from classes that are not installed here (``neural_wrappers.*``) or
whose pickled layout is older than the installed PyTorch (``ReduceLROnPlateau``, ``Adam``).

Only the tensors are needed. During unpickling every missing or incompatible class is
replaced by an inert stand-in (``_InertStub``), so the unpickler can finish and hand back
the ``"weights"`` list. This needs ``torch.load(..., weights_only=False)``, so only point
it at the official checkpoint files.

Every substitution (``sys.modules`` stubs, ``torch.optim.*`` patches, an ``__main__``
stand-in) is undone by ``_legacy_unpickling_scope`` in a ``finally`` block, even on error,
so the real ``torch.optim.AdamW`` used for training is never left patched. Do not widen
the scope beyond the single ``torch.load`` call.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from ._safeuav_net import ModelUNetDilatedConv, ModelUNetTinySum

#: Tensor / parameter count of the official HVO_small.pkl checkpoint.
EXPECTED_TENSOR_COUNT = 38
EXPECTED_PARAM_COUNT = 1_029_283
#: Positional indices of the final classifier layer's weight/bias within the 38-tensor
#: list -- the only two tensors NOT transplanted when adapting 3-class HVO to 2-class
#: SLZ (see build_binary_state_dict below).
FINAL_LAYER_INDICES = (36, 37)

#: Same, for HVO_large.pkl (ModelUNetDilatedConv, numFilters=64); the reference model's
#: parameter count matches the checkpoint's "weights" list exactly.
EXPECTED_TENSOR_COUNT_LARGE = 44
EXPECTED_PARAM_COUNT_LARGE = 23_895_107
FINAL_LAYER_INDICES_LARGE = (42, 43)

#: (architecture key, numFilters) by checkpoint filename -- explicit, not inferred
#: from the tensor shapes.
ARCHITECTURE_BY_CHECKPOINT_NAME = {
    "HVO_small.pkl": ("tiny_sum", 16),
    "HVO_large.pkl": ("dilated_conv", 64),
}


def _build_reference_model(architecture: str, num_filters: int, d_in: int, d_out: int) -> nn.Module:
    if architecture == "tiny_sum":
        return ModelUNetTinySum(dIn=d_in, dOut=d_out, numFilters=num_filters)
    if architecture == "dilated_conv":
        return ModelUNetDilatedConv(dIn=d_in, dOut=d_out, numFilters=num_filters)
    raise ValueError(f"unknown SafeUAV architecture: {architecture!r}")

_DUNDER_NAMES = frozenset({
    "__file__", "__path__", "__loader__", "__spec__", "__package__",
    "__all__", "__builtins__", "__cached__", "__name__",
})


class _InertStub:
    """A harmless placeholder for a class this loader doesn't need to actually run.

    No real ``__init__``/``__setstate__`` logic; any missing-attribute access (e.g. a
    legacy pickled bound method calling ``getattr(obj, name)`` at unpickle time)
    resolves to a no-op callable instead of raising, so reconstructing surrounding
    optimizer/scheduler/callback structures never crashes on an incompatible legacy
    attribute. This never executes any of the original class's real behavior.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def __repr__(self):
        return f"<InertStub {type(self).__module__}.{type(self).__name__}>"


class _AutoStubModule(types.ModuleType):
    """A module that fabricates an inert stub class for any *non-dunder* attribute
    accessed on it. Dunder attributes (``__file__``, ``__path__``, ...) raise
    AttributeError normally instead of being fabricated -- Python's own import/inspect
    machinery probes these on arbitrary sys.modules entries, and returning a fake `type`
    object for e.g. `__file__` breaks unrelated code (observed: torchvision import
    failing via `inspect.getframeinfo` when a stub module leaked into a later import)."""

    def __getattr__(self, name):
        if name in _DUNDER_NAMES or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        cls = type(name, (_InertStub,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


@contextlib.contextmanager
def _legacy_unpickling_scope():
    """Temporarily neutralize every legacy/incompatible class the official checkpoint
    may embed, for the duration of the ``with`` block only -- everything is restored to
    its original state afterward, success or failure, so no other code in this process
    (in particular, any later real ``torch.optim.AdamW``/scheduler usage) is affected."""
    restore_sys_modules: dict[str, object | None] = {}
    restore_attrs: list[tuple[object, str, object]] = []

    def install_stub_package(name: str) -> None:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            pkg_name = ".".join(parts[:i])
            if pkg_name not in sys.modules:
                restore_sys_modules.setdefault(pkg_name, None)  # None = "didn't exist"
                sys.modules[pkg_name] = _AutoStubModule(pkg_name)
            if i > 1:
                parent = ".".join(parts[: i - 1])
                parent_mod = sys.modules[parent]
                attr = parts[i - 1]
                if (parent_mod, attr) not in [(o, a) for o, a, _ in restore_attrs]:
                    restore_attrs.append((parent_mod, attr, getattr(parent_mod, attr, None)))
                setattr(parent_mod, attr, sys.modules[pkg_name])

    def patch_class(mod, name: str) -> None:
        obj = getattr(mod, name)
        if isinstance(obj, type):
            restore_attrs.append((mod, name, obj))
            setattr(mod, name, type(name, (_InertStub,), {"__module__": obj.__module__}))

    try:
        for pkg in (
            "neural_wrappers",
            "neural_wrappers.callbacks",
            "neural_wrappers.pytorch",
            "neural_wrappers.readers",
            "neural_wrappers.utilities",
        ):
            if pkg not in sys.modules:
                install_stub_package(pkg)

        # SafeUAV's own main.py defines a top-level `SchedulerCallback`, pickled with
        # module="__main__". Provide a temporary inert stand-in in this process's
        # __main__, restored afterward.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            had_attr = hasattr(main_mod, "SchedulerCallback")
            restore_attrs.append(
                (main_mod, "SchedulerCallback",
                 getattr(main_mod, "SchedulerCallback", None) if had_attr else None)
            )
            setattr(
                main_mod, "SchedulerCallback",
                type("SchedulerCallback", (_InertStub,), {"__module__": "__main__"}),
            )

        # Real torch.optim.* classes exist in this env, but their saved legacy internal
        # state (pre-1.7 PyTorch) does not match the currently-installed implementation
        # (confirmed empirically: ReduceLROnPlateau's pickled bound method `is_better =
        # self._cmp`, Adam's legacy `defaults` layout). Temporarily replace every
        # currently-known torch.optim.* class, in every torch.optim submodule (the
        # pickled GLOBAL opcode names the defining submodule, e.g.
        # "torch.optim.adam.Adam" -- patching only the `torch.optim` package-level
        # re-export is not enough), with an inert stub -- restored immediately after.
        import torch.optim as optim_pkg

        for _finder, mod_name, _is_pkg in pkgutil.iter_modules(
            optim_pkg.__path__, prefix="torch.optim."
        ):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            for name in list(vars(mod)):
                patch_class(mod, name)
        for name in list(vars(optim_pkg)):
            patch_class(optim_pkg, name)

        yield
    finally:
        for obj, name, original in reversed(restore_attrs):
            if original is None and not hasattr(type(obj), name):
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, original)
        for name, original in restore_sys_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _extract_weights_list(checkpoint_path: Path) -> List[torch.Tensor]:
    """Safely unpickle a SafeUAV/neural-wrappers checkpoint and return its raw,
    positional ``"weights"`` tensor list. Requires ``weights_only=False`` (see module
    docstring) -- only call this on an already-trusted, user-provided checkpoint path.
    All environment substitutions used to do so are scoped and reverted immediately
    (see ``_legacy_unpickling_scope``)."""
    with _legacy_unpickling_scope():
        obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "weights" not in obj:
        raise ValueError(
            f"{checkpoint_path}: expected a dict with a 'weights' key "
            f"(neural-wrappers checkpoint format), got {type(obj)!r} "
            f"with keys {list(obj.keys()) if isinstance(obj, dict) else 'n/a'}"
        )
    weights = obj["weights"]
    if not isinstance(weights, list) or not all(torch.is_tensor(t) for t in weights):
        raise ValueError(f"{checkpoint_path}: 'weights' is not a plain list of tensors")
    return weights


def load_pretrained_hvo3_state_dict(
    checkpoint_path: str | Path,
    *,
    architecture: str | None = None,
    num_filters: int | None = None,
    d_in: int = 3,
) -> "OrderedDict[str, torch.Tensor]":
    """Load the official 3-class (HVO) checkpoint into a name-keyed state_dict.

    ``architecture``/``num_filters`` default to ``None``, meaning "look up
    ``ARCHITECTURE_BY_CHECKPOINT_NAME`` by the checkpoint's own filename" (``tiny_sum``/
    16 for ``HVO_small.pkl``, ``dilated_conv``/64 for ``HVO_large.pkl``) — pass them
    explicitly only to override that lookup (e.g. a renamed copy of the same file).

    Reproduces the official ``NetworkSerializer.doLoadWeights`` positional-copy
    contract exactly (same two assertions: total scalar-parameter-count match, then
    per-tensor shape match) against a freshly-built, faithfully-ported ``dOut=3``
    architecture, then returns that model's own (name-keyed) ``state_dict()`` — this
    is the artifact the binary-head adaptation (``build_binary_state_dict``) transplants
    from.
    """
    checkpoint_path = Path(checkpoint_path)
    weights = _extract_weights_list(checkpoint_path)

    if architecture is None or num_filters is None:
        default_arch, default_nf = ARCHITECTURE_BY_CHECKPOINT_NAME.get(
            checkpoint_path.name, ("tiny_sum", 16)
        )
        architecture = architecture or default_arch
        num_filters = num_filters if num_filters is not None else default_nf

    ref_model = _build_reference_model(architecture, num_filters, d_in=d_in, d_out=3)
    params = [p for p in ref_model.parameters() if p.requires_grad]

    loaded_count = sum(t.numel() for t in weights)
    this_count = sum(p.numel() for p in params)
    if loaded_count != this_count:
        raise ValueError(
            f"{checkpoint_path}: total parameter count mismatch "
            f"(checkpoint={loaded_count}, reconstructed dOut=3 model={this_count})"
        )
    if len(weights) != len(params):
        raise ValueError(
            f"{checkpoint_path}: tensor count mismatch "
            f"(checkpoint={len(weights)}, reconstructed dOut=3 model={len(params)})"
        )
    for i, (p, t) in enumerate(zip(params, weights)):
        if tuple(p.shape) != tuple(t.shape):
            raise ValueError(
                f"{checkpoint_path}: shape mismatch at position {i}: "
                f"checkpoint={tuple(t.shape)} vs reconstructed={tuple(p.shape)}"
            )
        with torch.no_grad():
            p.copy_(t)

    return ref_model.state_dict()


def build_binary_state_dict(
    hvo3_state_dict: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """Transplant every pretrained tensor except ``finalConv.{weight,bias}`` into a
    fresh, name-keyed state_dict suitable for a ``dOut=2`` model — the 2-class head
    itself is left absent (the caller's ``dOut=2`` model keeps its own random init for
    those two tensors after a ``strict=False`` load of the returned dict)."""
    return OrderedDict(
        (k, v) for k, v in hvo3_state_dict.items() if not k.startswith("finalConv.")
    )
