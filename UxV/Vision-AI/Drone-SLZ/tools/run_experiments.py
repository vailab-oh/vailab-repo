#!/usr/bin/env python3
"""Plan, and optionally run, every training and scoring run of the paper.

**Default = dry run**: prints a ``model | size | fold | seed | lr / wd | status`` table and
runs nothing. Add ``--execute`` to launch the runs one at a time, in order.

Modes (``--mode``):
  train         every model x 4 folds x 3 seeds with the one recipe of
                ``slz_benchmark.experiments``, launched with torchrun
                -> ``<exp dir>/main/<model>/fold<N>/seed<S>/``
  eval          score each finished ``train`` run once on its fold's test set
                (evaluate.py --split test) -> ``<run dir>/test_metrics.json``. Unfinished
                runs are shown as ``training`` and never scored. Tables:
                tools/summarize_results.py
  radius-train  footprint-radius study: the base size of each family x 4 folds x seed 42 on
                each label set of ``experiments.RADIUS_VARIANTS`` (``<--root>_<tag>``, from
                tools/make_radius_variant.py) -> ``<exp dir>/radius/<tag>/<model>/...``.
                ``--root`` is the BASE dataset root.
  radius-eval   score each finished radius-train run on its own radius' test labels.
                Tables: tools/summarize_radius.py

Experiment directory: ``--exp-dir``, else ``outputs/experiments_<today>`` for a new
``--mode train`` and the latest ``outputs/experiments_*`` for every other mode and for
``--skip-existing`` (so a resumed or scored queue finds its runs).

Filters (combinable): ``--model`` (keys or family prefixes, e.g. ``segformer_b5`` or
``upernet``; default all), ``--fold 1``, ``--seed 42``, ``--radius r050`` / ``0.5``.

A failed run is logged and the queue continues (``--stop-on-error`` aborts instead); each
run's console output also goes to ``<exp dir>/logs/``. ``--skip-existing`` resumes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from slz_benchmark import experiments as E

TRAIN = REPO / "train.py"
EVAL = REPO / "evaluate.py"


def _finished(out_dir: Path) -> bool:
    """Finished = ``history.json`` has ``config.json["epochs"]`` entries. Not merely
    ``best_metrics.json`` existing: the trainer writes it after epoch 1."""
    try:
        epochs = json.loads((out_dir / "config.json").read_text())["epochs"]
        done = len(json.loads((out_dir / "history.json").read_text()))
    except (OSError, ValueError, KeyError):
        return False
    return done >= epochs and (out_dir / "best_metrics.json").exists()


def _eval_done(path: Path) -> bool:
    return path.exists()


# --------------------------------------------------------------------------- plans
def _train_cmd_prefix(nproc_per_node: int) -> list[str]:
    """Training always goes through torchrun; --nproc_per_node=1 is a single process."""
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}", str(TRAIN)]


def _train_cmd(root: Path, out: Path, model: str, fold: int, seed: int,
               nproc_per_node: int, scheme: str = E.SPLIT_SCHEME) -> list[str]:
    return [
        *_train_cmd_prefix(nproc_per_node),
        "--root", str(root), "--output", str(out),
        "--model", model,
        "--scheme", scheme, "--fold", str(fold),
        "--seed", str(seed),
        "--epochs", str(E.EPOCHS),
        "--batch-size", str(E.GLOBAL_BATCH_SIZE),  # global; train.py divides it per rank
        "--lr", str(E.LR), "--weight-decay", str(E.WEIGHT_DECAY),
        "--jitter-prob", repr(E.JITTER_PROB),
        "--blur-prob", repr(E.BLUR_PROB),
        "--eval-split", "none",  # test is scored once, afterwards, by --mode eval
    ]


def _row(model: str, fold: int, seed: int, out: Path, cmd: list[str], nproc_per_node: int) -> dict:
    return {
        "phase": "train", "model": model, "model_label": E.family(model),
        "size_label": E.size(model), "is_base": E.is_base(model),
        "fold": fold, "seed": seed, "lr": E.LR, "wd": E.WEIGHT_DECAY,
        "epochs": E.EPOCHS, "global_batch_size": E.GLOBAL_BATCH_SIZE,
        "nproc_per_node": nproc_per_node,
        "jitter_prob": E.JITTER_PROB, "blur_prob": E.BLUR_PROB,
        "safe_weight": "auto (per-fold inverse frequency)",
        "out_dir": str(out), "cmd": cmd, "kind": "train",
    }


def plan_train(root: Path, exp_dir: Path, nproc_per_node: int = E.NPROC_PER_NODE) -> list[dict]:
    """Every model x fold x seed: the four base sizes first, then the larger sizes."""
    runs = []
    for model in E.queue_order():
        for fold in E.FOLDS:
            for seed in E.SEEDS:
                out = E.run_dir(exp_dir, model, fold, seed)
                runs.append(_row(model, fold, seed, out,
                                 _train_cmd(root, out, model, fold, seed, nproc_per_node),
                                 nproc_per_node))
    return runs


def _eval_row(t: dict, root: Path, phase: str, keys: tuple) -> dict:
    run_dir = Path(t["out_dir"])
    save = run_dir / "test_metrics.json"
    cmd = [
        sys.executable, str(EVAL),
        "--root", str(root), "--checkpoint", str(run_dir / "best_model"),
        "--model", t["model"],
        "--scheme", E.SPLIT_SCHEME, "--fold", str(t["fold"]),
        "--split", "test",
        "--save-json", str(save),
    ]
    r = {k: t[k] for k in keys}
    r.update({"phase": phase, "out_dir": str(save), "requires": str(run_dir),
              "cmd": cmd, "kind": "eval"})
    return r


_EVAL_KEYS = ("model", "model_label", "size_label", "is_base", "fold", "seed", "lr", "wd")


def plan_eval(root: Path, exp_dir: Path) -> list[dict]:
    """Test scoring of the ``train`` queue, one row per training run. A row whose training
    is not finished is never executed. The adapter rebuilds the architecture from the
    checkpoint itself."""
    return [_eval_row(t, root, "eval", _EVAL_KEYS) for t in plan_train(root, exp_dir, 1)]


def _radius_root(root: Path, tag: str) -> Path:
    """The radius-variant dataset root; fails fast if it has no class weights (train.py
    would otherwise train with safe_weight = 1.0)."""
    vroot = E.radius_variant_root(root, tag)
    cw = vroot / "manifests" / E.SPLIT_SCHEME / "class_weights.json"
    if not cw.exists():
        raise SystemExit(
            f"radius variant {tag}: {cw} not found. Build it with\n"
            f"  python tools/make_radius_variant.py --src-root {root} --uav-radius-m <r>\n"
            f"  python prepare_data.py --stage splits --root {vroot.parent.parent} "
            f"--dataset-name {Path(root).name} --variant {tag}")
    return vroot


def plan_radius_train(root: Path, exp_dir: Path,
                      nproc_per_node: int = E.NPROC_PER_NODE) -> list[dict]:
    """For each ``experiments.RADIUS_VARIANTS`` label set: each base size x 4 folds x
    ``experiments.RADIUS_SEED``, same recipe; only ``--root`` points at the variant."""
    runs = []
    for tag, radius, pca_radius in E.RADIUS_VARIANTS:
        vroot = _radius_root(root, tag)
        for model in E.BASE_MODELS:
            for fold in E.FOLDS:
                seed = E.RADIUS_SEED
                out = E.radius_run_dir(exp_dir, tag, model, fold, seed)
                t = _row(model, fold, seed, out,
                         _train_cmd(vroot, out, model, fold, seed, nproc_per_node),
                         nproc_per_node)
                t.update({"phase": "radius_train", "radius_tag": tag, "radius_m": radius,
                          "pca_radius_m": pca_radius,
                          "size_label": f"{t['size_label']} r{radius:.2f}/p{pca_radius:.2f}",
                          "dataset_root": str(vroot)})
                runs.append(t)
    return runs


def plan_radius_eval(root: Path, exp_dir: Path) -> list[dict]:
    """Test scoring of each finished ``radius-train`` run on its own radius' labels."""
    keys = _EVAL_KEYS + ("radius_tag", "radius_m", "pca_radius_m", "dataset_root")
    return [_eval_row(t, Path(t["dataset_root"]), "radius_eval", keys)
            for t in plan_radius_train(root, exp_dir, 1)]


PLANNERS = {
    "train": plan_train,
    "eval": plan_eval,
    "radius-train": plan_radius_train,
    "radius-eval": plan_radius_eval,
}
_DDP_MODES = {"train", "radius-train"}


# ------------------------------------------------------------------- filters / printing
def _model_matches(model: str, wanted) -> bool:
    """``wanted`` holds keys (``segformer_b5``) or family prefixes (``segformer``)."""
    return any(model == w or model.startswith(w + "_") for w in wanted)


def _matches(r: dict, args) -> bool:
    if args.model is not None and not _model_matches(r["model"], args.model):
        return False
    if args.fold is not None and r.get("fold") != args.fold:
        return False
    if args.seed is not None and r.get("seed") != args.seed:
        return False
    if args.radius is not None:
        want = args.radius.lower()
        tag = r.get("radius_tag")
        if tag is None or want not in (tag, f"{r['radius_m']:g}", f"{r['radius_m']:.2f}"):
            return False
    return True


def _fmt_lr(lr: float) -> str:
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def _is_done(r: dict) -> bool:
    return _finished(Path(r["out_dir"])) if r["kind"] == "train" else _eval_done(Path(r["out_dir"]))


def _ready(r: dict) -> bool:
    """False for an ``eval`` row whose training run is not finished yet."""
    return "requires" not in r or _finished(Path(r["requires"]))


def _status(r: dict, done: bool) -> str:
    if done:
        if r["kind"] == "eval":
            try:
                miou = json.loads(Path(r["out_dir"]).read_text())["metrics"]["miou"]
                return f"DONE {miou:.3f}"
            except (OSError, ValueError, KeyError, TypeError):
                return "DONE"
        return "DONE"
    return "todo" if _ready(r) else "training"


def _print_table(runs: list[dict], done: list[bool], verbose: bool) -> None:
    """Per-(model, size) summary, then one row per run."""
    groups: dict = {}
    for r, d in zip(runs, done):
        g = groups.setdefault((r["model_label"], r["size_label"]),
                              {"folds": set(), "seeds": set(), "done": 0, "n": 0})
        g["folds"].add(r["fold"]); g["seeds"].add(r["seed"]); g["n"] += 1; g["done"] += d
    print(f"\n  {'model':<12} {'size':<12} {'folds':>5} {'seeds':>6} {'done/total':>11}")
    print("  " + "-" * 50)
    for (m, sz), g in groups.items():
        print(f"  {m:<12} {sz:<12} {len(g['folds']):>5} {len(g['seeds']):>6} "
              f"{g['done']:>5}/{g['n']:<5}")
    print(f"\n  {'#':>3}  {'model':<12} {'size':<12} {'fold':>4} {'seed':>5}  {'lr / wd':<14} status")
    print("  " + "-" * 62)
    for i, (r, d) in enumerate(zip(runs, done), 1):
        hp = _fmt_lr(r["lr"]) + " / " + format(r["wd"], "g")
        print(f"  {i:>3}  {r['model_label']:<12} {r['size_label']:<12} {r['fold']:>4} {r['seed']:>5}  "
              f"{hp:<14} {_status(r, d)}")
        if verbose:
            print(f"       out: {r['out_dir']}")
            print("       cmd: " + " ".join(r["cmd"]))


# ------------------------------------------------------------------------- execution
def _run_id(r: dict, exp_dir: Path) -> str:
    p = Path(r["out_dir"])
    if p.suffix == ".json":
        p = p.with_suffix("")
    try:
        return "_".join(p.relative_to(exp_dir).parts)
    except ValueError:
        return p.name


def _run_logged(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd``, mirroring its combined stdout/stderr (raw bytes, so tqdm's ``\\r``
    progress bars survive) to the console AND to ``log_path``. Returns the exit code."""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
                log.write(chunk); log.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise


def _execute(mode: str, to_execute: list[dict], exp_dir: Path, stop_on_error: bool) -> int:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = exp_dir / "logs" / f"{mode}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nlogs: {log_dir}")
    failures: list[tuple[int, dict, int, Path]] = []
    for i, r in enumerate(to_execute, 1):
        rid = _run_id(r, exp_dir)
        log_path = log_dir / f"{i:03d}_{rid}.log"
        print(f"\n>>> [{i}/{len(to_execute)}] {rid}\n    {' '.join(r['cmd'])}\n")
        rc = _run_logged(r["cmd"], log_path)
        if rc != 0:
            failures.append((i, r, rc, log_path))
            print(f"\n!!! [{i}/{len(to_execute)}] FAILED (exit {rc}): {rid}\n    log: {log_path}")
            if stop_on_error:
                print("    --stop-on-error: aborting the queue.")
                break
    print("\n" + "=" * 88)
    print(f"finished: {len(to_execute) - len(failures)} ok / {len(failures)} failed "
          f"/ {len(to_execute)} planned")
    for i, r, rc, lp in failures:
        print(f"  FAILED [{i}] {_run_id(r, exp_dir)} (exit {rc})  log: {lp}")
    if failures:
        print("Re-run the same command with --skip-existing to retry only the unfinished runs.")
    print("=" * 88)
    return 1 if failures else 0


def resolve_exp_dir(exp_dir, mode: str, skip_existing: bool) -> Path:
    """``--exp-dir`` if given; a new dated directory for a fresh ``--mode train``; else the
    latest ``outputs/experiments_*`` (falling back to a new one for ``radius-train``)."""
    if exp_dir is not None:
        return Path(exp_dir).resolve()
    if mode == "train" and not skip_existing:
        return E.new_experiment_dir()
    latest = E.latest_experiment_dir()
    if latest is not None:
        return latest
    if mode in _DDP_MODES:
        return E.new_experiment_dir()
    raise SystemExit(f"no {E.OUTPUTS}/{E.EXPERIMENTS_PREFIX}* directory yet; "
                     "run --mode train first or pass --exp-dir")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="prepared dataset root (the BASE root for the radius modes)")
    ap.add_argument("--mode", required=True, choices=list(PLANNERS))
    ap.add_argument("--exp-dir", type=Path, default=None,
                    help="experiment directory (default: see the module docstring)")
    ap.add_argument("--nproc-per-node", type=int, default=E.NPROC_PER_NODE,
                    help="DDP world size of the training modes (torchrun --nproc_per_node); "
                         "1 = single GPU. Ignored by the eval modes.")
    ap.add_argument("--execute", action="store_true",
                    help="actually launch the runs (default: print only)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose output already looks finished")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort the queue at the first failed run (default: log it, "
                         "continue, and list all failures at the end)")
    ap.add_argument("--model", nargs="+", default=None,
                    help="only these models: keys (segformer_b5) or family prefixes "
                         f"(segformer, deeplab, upernet, safeuav). Keys: {', '.join(E.MODELS)}")
    ap.add_argument("--fold", type=int, default=None, help="only this fold (1-4)")
    ap.add_argument("--seed", type=int, default=None, help="only this seed")
    ap.add_argument("--radius", default=None,
                    help="radius modes only: this variant, by tag (r050) or value (0.5)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print each run's output dir and full command")
    args = ap.parse_args()

    exp_dir = resolve_exp_dir(args.exp_dir, args.mode, args.skip_existing)
    filtering = any(v is not None for v in (args.model, args.fold, args.seed, args.radius))
    if args.mode in _DDP_MODES:
        planned = PLANNERS[args.mode](args.root, exp_dir, args.nproc_per_node)
    else:
        planned = PLANNERS[args.mode](args.root, exp_dir)
    runs = [r for r in planned if _matches(r, args)]
    done = [_is_done(r) for r in runs]
    print("\n" + "=" * 88)
    print(f"MODE {args.mode}  —  {len(runs)} run(s)  —  {exp_dir}")
    print("=" * 88)
    _print_table(runs, done, args.verbose)

    if filtering and not runs:
        have = lambda k: sorted({str(r[k]) for r in planned if k in r})
        ap.error("the --model/--fold/--seed/--radius filter matched no run.\n"
                 f"  models: {have('model')}\n"
                 f"  folds : {have('fold')}\n  seeds : {have('seed')}"
                 + (f"\n  radii : {have('radius_tag')}" if have('radius_tag') else ""))

    to_execute, blocked = [], 0
    for r, d in zip(runs, done):
        if args.skip_existing and d:
            continue
        if not _ready(r):
            blocked += 1
            continue
        to_execute.append(r)

    print("\n" + "=" * 88)
    print(f"planned: {len(runs)} run(s) | to execute: {len(to_execute)} | "
          f"execute={args.execute}"
          + (f" | blocked (training not finished): {blocked}" if blocked else ""))
    print("=" * 88)

    if not args.execute:
        print("\nDRY RUN — nothing executed. Re-run with --execute to launch.")
        return
    sys.exit(_execute(args.mode, to_execute, exp_dir, args.stop_on_error))


if __name__ == "__main__":
    main()
