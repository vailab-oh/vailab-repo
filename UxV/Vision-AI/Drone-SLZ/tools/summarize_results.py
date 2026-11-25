#!/usr/bin/env python3
"""Aggregate the test results of the training runs into the paper tables.

    python tools/summarize_results.py      # latest outputs/experiments_*, -> <exp dir>/results/

Reads ``<run dir>/test_metrics.json`` (written by ``run_experiments.py --mode eval``); the
run list and directories come from ``run_experiments.plan_train``. The FOLD is the
experimental unit (n = 4); the seeds of a fold are replicates of the same held-out
sequence, not independent samples.

Outputs (``--out``, default ``<exp dir>/results/``):
  runs.csv      one row per scored run (model, size, fold, seed, six metrics, frames, checkpoint)
  per_fold.csv  (model, size, fold): mean over seeds and seed std (empty with a single seed)
  summary.csv   (model, size): overall mean = mean of the fold means; cross-scene std = std of
                the fold means; plus the mean seed std
  pairwise.csv  base-size models only, per pair on --metric: per-fold differences d_f,
                mean(d), SE = std(d)/sqrt(n), mean(d)/std(d), wins k/n. No t-test / Wilcoxon
                (pseudo-replication: seeds of one fold are not independent)
  size_pairwise.csv  the same, each larger size vs the base size of its own model
  summary.md    the same as paper-style tables (main table, size table, per-fold
                supplementary, pairwise), with the conventions in the captions

All standard deviations are sample standard deviations (ddof = 1). Runs without a
test_metrics.json are reported as missing and left out; a (model, size) with fewer than 4
folds is marked ``(n/4 folds)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import run_experiments as RE  # noqa: E402  (planner only; its main() is not run)
from slz_benchmark import experiments as E  # noqa: E402

METRICS = [
    ("miou", "mIoU"),
    ("iou_safe", "IoU Safe"),
    ("iou_unsafe", "IoU Unsafe"),
    ("precision_safe", "Precision Safe"),
    ("recall_safe", "Recall Safe"),
    ("f1_safe", "F1 Safe"),
]
N_FOLDS = len(E.FOLDS)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else None


def _f(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def collect(exp_dir: Path):
    """-> (scored rows, missing rows). Each row is the plan_train dict + 'metrics' etc."""
    scored, missing = [], []
    for t in RE.plan_train(Path("."), exp_dir, 1):
        p = Path(t["out_dir"]) / "test_metrics.json"
        if not p.exists():
            missing.append(t)
            continue
        d = json.loads(p.read_text())
        scored.append({**t, "metrics": d["metrics"], "frames": d.get("frames"),
                       "checkpoint": d.get("checkpoint"), "json": str(p)})
    return scored, missing


def group(scored):
    """model -> {"label":..., "is_base":..., "folds": {fold: [row, ...]}}, plan order."""
    out = {}
    for r in scored:
        g = out.setdefault(r["model"], {
            "model_label": r["model_label"], "size_label": r["size_label"],
            "is_base": r["is_base"], "folds": {}})
        g["folds"].setdefault(r["fold"], []).append(r)
    return out


def per_fold_stats(g):
    """fold -> {metric: (mean over seeds, seed sd, n seeds)}"""
    res = {}
    for fold in sorted(g["folds"]):
        rows = g["folds"][fold]
        res[fold] = {}
        for k, _ in METRICS:
            xs = [r["metrics"][k] for r in rows]
            res[fold][k] = (_mean(xs), _sd(xs), len(xs))
    return res


def summary_stats(pf):
    """metric -> (overall mean, cross-scene sd, mean seed sd, n folds)"""
    res = {}
    for k, _ in METRICS:
        fm = [pf[f][k][0] for f in pf]
        seed_sds = [pf[f][k][1] for f in pf if pf[f][k][1] is not None]
        res[k] = (_mean(fm), _sd(fm), _mean(seed_sds), len(fm))
    return res


def _base_pairs(groups):
    base = [g for g in groups.values() if g["is_base"]]
    return [(base[i], base[j]) for i in range(len(base)) for j in range(i + 1, len(base))]


def _size_pairs(groups):
    """(larger size, base size of the same family): does the bigger model help?"""
    base = {m: g for m, g in groups.items() if g["is_base"]}
    return [(g, base[E.base_of(m)]) for m, g in groups.items()
            if not g["is_base"] and E.base_of(m) in base]


def pairwise(groups, metric, pairs=_base_pairs):
    rows = []
    for ga, gb in pairs(groups):
        pa, pb = per_fold_stats(ga), per_fold_stats(gb)
        folds = sorted(set(pa) & set(pb))
        d = [pa[f][metric][0] - pb[f][metric][0] for f in folds]
        if not d:
            continue
        sd = _sd(d)
        rows.append({
            "a": ga["model_label"] + " " + ga["size_label"],
            "b": gb["model_label"] + " " + gb["size_label"],
            "folds": folds, "d": d, "mean_d": _mean(d), "sd_d": sd,
            "se": sd / math.sqrt(len(d)) if sd is not None else None,
            "std_effect": (_mean(d) / sd) if sd not in (None, 0) else None,
            "wins": sum(x > 0 for x in d), "n": len(d),
        })
    return rows


def write_csvs(out: Path, scored, groups, pw, spw=()):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "size", "fold", "seed", *[k for k, _ in METRICS], "pixel_acc",
                    "frames", "checkpoint"])
        for r in scored:
            w.writerow([r["model_label"], r["size_label"], r["fold"], r["seed"],
                        *[_f(r["metrics"][k]) for k, _ in METRICS],
                        _f(r["metrics"].get("pixel_acc")), r["frames"], r["checkpoint"]])
    with open(out / "per_fold.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "size", "fold", "n_seeds",
                    *[f"{k}_{s}" for k, _ in METRICS for s in ("mean", "seed_sd")]])
        for g in groups.values():
            pf = per_fold_stats(g)
            for fold, st in pf.items():
                w.writerow([g["model_label"], g["size_label"], fold, st["miou"][2],
                            *[_f(v) for k, _ in METRICS for v in st[k][:2]]])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "size", "base_size", "n_folds",
                    *[f"{k}_{s}" for k, _ in METRICS for s in ("mean", "cross_scene_sd", "mean_seed_sd")]])
        for g in groups.values():
            sm = summary_stats(per_fold_stats(g))
            w.writerow([g["model_label"], g["size_label"], int(g["is_base"]), sm["miou"][3],
                        *[_f(v) for k, _ in METRICS for v in sm[k][:3]]])
    for name, rows in (("pairwise.csv", pw), ("size_pairwise.csv", spw)):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "folds", "d_per_fold (a - b)", "mean_d", "sd_d", "se", "mean_d/sd_d",
                        "wins_a", "n_folds"])
            for p in rows:
                w.writerow([p["a"], p["b"], " ".join(map(str, p["folds"])),
                            " ".join(_f(x) for x in p["d"]), _f(p["mean_d"]), _f(p["sd_d"]),
                            _f(p["se"]), _f(p["std_effect"], 2), p["wins"], p["n"]])


def _cell(st, nd=3):
    mean, sd, _, n = st
    s = _f(mean, nd) + (f" ({_f(sd, nd)})" if sd is not None else "")
    return s + (f" ({n}/{N_FOLDS} folds)" if n < N_FOLDS else "")


def write_md(out: Path, groups, pw, missing, metric, spw=()):
    L = ["# SLZ test results", ""]
    L += ["Generated by `tools/summarize_results.py` from `<run dir>/test_metrics.json`. "
          "Fold is the experimental unit (n = 4 held-out scenes); seeds within a fold are "
          "replicates. All std are sample std (ddof = 1).", ""]
    base = [g for g in groups.values() if g["is_base"]]
    big = [g for g in groups.values() if not g["is_base"]]
    head = "| Model | " + " | ".join(lbl for _, lbl in METRICS) + " |"
    sep = "|---|" + "---:|" * len(METRICS)
    if base:
        seed_sd = _mean([summary_stats(per_fold_stats(g))["miou"][2] for g in base
                         if summary_stats(per_fold_stats(g))["miou"][2] is not None])
        L += ["## Main table (base size of each model)", "",
              "Cell: overall mean over folds (cross-scene std = std of the 4 fold means). "
              + (f"Mean seed std of mIoU within a fold: {_f(seed_sd, 3)}." if seed_sd is not None else ""),
              "", head, sep]
        for g in base:
            sm = summary_stats(per_fold_stats(g))
            L.append(f"| {g['model_label']} {g['size_label']} | "
                     + " | ".join(_cell(sm[k]) for k, _ in METRICS) + " |")
        L.append("")
    if big:
        n_big = sorted({len(rows) for g in big for rows in g["folds"].values()})
        L += ["## Larger sizes", "",
              "Cell: mean over folds (cross-scene std). Seeds per (size, fold): "
              + "/".join(map(str, n_big))
              + ". " + (lambda v: f"Mean seed std of mIoU within a fold: {_f(v, 3)}." if v is not None
                        else "One seed: no seed std -- compare against the base size's seed std above.")(
                  _mean([x for x in (summary_stats(per_fold_stats(g))["miou"][2] for g in big) if x is not None])),
              "",
              "| Model | Size | " + " | ".join(lbl for _, lbl in METRICS) + " |",
              "|---|---|" + "---:|" * len(METRICS)]
        for g in big:
            sm = summary_stats(per_fold_stats(g))
            L.append(f"| {g['model_label']} | {g['size_label']} | "
                     + " | ".join(_cell(sm[k]) for k, _ in METRICS) + " |")
        L.append("")
    if groups:
        L += ["## Per fold (supplementary)", "",
              "Cell: mean ± seed std over the seeds of that fold.", "",
              "| Model | Size | Fold | mIoU | Precision Safe | Recall Safe | F1 Safe |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for g in groups.values():
            for fold, st in per_fold_stats(g).items():
                cells = []
                for k in ("miou", "precision_safe", "recall_safe", "f1_safe"):
                    m, sd, _ = st[k]
                    cells.append(_f(m, 3) + (f" ± {_f(sd, 3)}" if sd is not None else ""))
                L.append(f"| {g['model_label']} | {g['size_label']} | {fold} | " + " | ".join(cells) + " |")
        L.append("")
    name = dict(METRICS)[metric]
    for title, rows in ((f"Pairwise, base sizes ({name})", pw),
                        (f"Larger size vs base size of the same model ({name})", spw)):
        if not rows:
            continue
        L += [f"## {title}", "",
              "d_f = A(f) − B(f) from the per-fold seed means. SE = std(d)/√n. Effect = mean(d)/std(d). "
              "Wins = folds where A > B; 4/4 corresponds to a one-sided sign-test p = 1/16 = 0.0625, "
              "quoted descriptively, not as significance. No t-test / Wilcoxon on (fold, seed) rows.", "",
              "| A | B | d per fold | mean(d) ± SE | effect | wins (A>B) |", "|---|---|---|---:|---:|---:|"]
        for p in rows:
            L.append(f"| {p['a']} | {p['b']} | " + " ".join(_f(x, 3) for x in p["d"]) + " | "
                     + _f(p["mean_d"], 3) + (f" ± {_f(p['se'], 3)}" if p["se"] is not None else "")
                     + f" | {_f(p['std_effect'], 2)} | {p['wins']}/{p['n']} |")
        L.append("")
    L += [f"## Missing ({len(missing)} runs without test_metrics.json)", ""]
    if missing:
        by = {}
        for t in missing:
            by.setdefault(f"{t['model_label']} {t['size_label']}", []).append(f"f{t['fold']}s{t['seed']}")
        L += [f"- {k}: {', '.join(v)}" for k, v in by.items()]
    else:
        L.append("- none")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--exp-dir", type=Path, default=None,
                    help="experiment directory (default: the latest outputs/experiments_*)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <exp-dir>/results)")
    ap.add_argument("--metric", default="miou", choices=[k for k, _ in METRICS],
                    help="metric used for the pairwise comparison")
    args = ap.parse_args()
    exp_dir = args.exp_dir or E.latest_experiment_dir()
    if exp_dir is None:
        ap.error("no outputs/experiments_* directory; pass --exp-dir")
    out = args.out or exp_dir / "results"

    scored, missing = collect(exp_dir)
    groups = group(scored)
    pw = pairwise(groups, args.metric)
    spw = pairwise(groups, args.metric, pairs=_size_pairs)
    write_csvs(out, scored, groups, pw, spw)
    write_md(out, groups, pw, missing, args.metric, spw)

    print(f"scored runs: {len(scored)} | missing: {len(missing)}")
    for g in groups.values():
        sm = summary_stats(per_fold_stats(g))
        print(f"  {g['model_label']:<12} {g['size_label']:<12} mIoU {_cell(sm['miou'])}")
    print(f"written: {out}/{{runs,per_fold,summary,pairwise,size_pairwise}}.csv, summary.md")


if __name__ == "__main__":
    main()
