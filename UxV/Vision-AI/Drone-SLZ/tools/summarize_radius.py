#!/usr/bin/env python3
"""Tables for the UAV footprint-radius study (experiments.RADIUS_VARIANTS vs the base r).

Each variant changes only the UAV radius r (PCA radius kept at the base 0.50 m):

    python tools/summarize_radius.py --base-root <UAVSCENES>/prepared/uavscenes_slz

Read-only. Inputs (``--exp-dir``, default the latest ``outputs/experiments_*``):
  * ``<exp dir>/radius/<tag>/<model>/fold*/seed<RADIUS_SEED>/test_metrics.json``
    (``run_experiments.py --mode radius-eval``) -- each scored on ITS OWN radius' test labels;
  * ``<exp dir>/main/<model>/fold*/seed*/test_metrics.json`` -- the base
    r = RADIUS_BASE_M runs: seed RADIUS_SEED is the seed-matched comparison, all of
    SEEDS give the within-fold seed std used as the noise yardstick;
  * optional ``--base-root``: ``<base root>_<tag>/label_stats.csv`` (tools/make_radius_variant.py)
    + the split manifests, for the label-statistics table.

Same conventions as tools/summarize_results.py (fold = experimental unit, n = 4; sample
std, ddof = 1; no significance tests). The radius changes the TEST labels too, so a
difference between radii is "how well each label set can be learned and predicted", not
a ranking of models or of radii.

Outputs (``--out``, default ``<exp dir>/results_radius/``): runs.csv, summary.csv,
per_fold.csv, delta.csv, label_stats.csv (if --base-root), summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from summarize_results import METRICS, _f, _mean, _sd  # noqa: E402
from slz_benchmark import experiments as E  # noqa: E402

N_FOLDS = len(E.FOLDS)
BASE_TAG = f"r{round(E.RADIUS_BASE_M * 100):03d}"


def _variants():
    """experiments.RADIUS_VARIANTS sorted by radius (display order; the queue order is separate)."""
    return [(t, r) for t, r, _ in sorted(E.RADIUS_VARIANTS, key=lambda x: x[1])]


PCA_OF = {t: p for t, _, p in E.RADIUS_VARIANTS}


def _pca(tag: str) -> float:
    return PCA_OF.get(tag, E.RADIUS_BASE_PCA_M)


def _rl(tag: str, r: float) -> str:
    """column label 'r=0.20/0.35' = UAV radius / PCA radius"""
    return f"r={r:.2f}/{_pca(tag):.2f}"


def _radii():
    """[(tag, radius_m), ...] sorted by radius, the base radius included."""
    return sorted([*_variants(), (BASE_TAG, E.RADIUS_BASE_M)], key=lambda x: x[1])


def _metrics(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"]


def collect(exp_dir: Path):
    """-> rows (model, radius, fold, seed, metrics), missing list, seed-std table."""
    rows, missing = [], []
    seed_sd = {}  # (model, fold) -> {metric: sd over SEEDS} at the base radius
    for model in E.BASE_MODELS:
        for tag, r in _radii():
            for fold in E.FOLDS:
                if tag == BASE_TAG:
                    p = E.run_dir(exp_dir, model, fold, E.RADIUS_SEED) / "test_metrics.json"
                else:
                    p = E.radius_run_dir(exp_dir, tag, model, fold, E.RADIUS_SEED) / "test_metrics.json"
                m = _metrics(p)
                (rows if m else missing).append(
                    {"model": model, "model_label": E.family(model),
                     "size_label": E.size(model),
                     "tag": tag, "radius": r, "fold": fold, "seed": E.RADIUS_SEED,
                     "metrics": m, "json": str(p)})
        for fold in E.FOLDS:
            ms = [_metrics(E.run_dir(exp_dir, model, fold, s) / "test_metrics.json")
                  for s in E.SEEDS]
            ms = [m for m in ms if m]
            seed_sd[(model, fold)] = {k: _sd([m[k] for m in ms]) for k, _ in METRICS}
    return rows, missing, seed_sd


def _index(rows):
    return {(r["model"], r["tag"], r["fold"]): r["metrics"] for r in rows}


def summary(idx, model, tag):
    """metric -> (mean over folds, cross-scene sd, n folds)"""
    out = {}
    for k, _ in METRICS:
        xs = [idx[(model, tag, f)][k] for f in E.FOLDS if (model, tag, f) in idx]
        out[k] = (_mean(xs), _sd(xs), len(xs))
    return out


def deltas(idx, seed_sd, metric):
    out = []
    for model in E.BASE_MODELS:
        for tag, r in _variants():
            folds = [f for f in E.FOLDS if (model, tag, f) in idx and (model, BASE_TAG, f) in idx]
            d = [idx[(model, tag, f)][metric] - idx[(model, BASE_TAG, f)][metric] for f in folds]
            if not d:
                continue
            sd = _sd(d)
            ssd = [seed_sd[(model, f)][metric] for f in folds if seed_sd[(model, f)][metric] is not None]
            out.append({"model": model, "label": E.label(model),
                        "tag": tag, "radius": r, "folds": folds, "d": d, "mean_d": _mean(d),
                        "se": sd / math.sqrt(len(d)) if sd is not None else None,
                        "n_up": sum(x > 0 for x in d), "n": len(d),
                        "base_seed_sd": _mean(ssd)})
    return out


def label_stats(base_root: Path):
    """Per held-out family (= that fold's test frames) and over the whole LOSO pool:
    Safe fraction of labeled px per radius, % labeled px changed vs base, frames with no Safe."""
    import pandas as pd

    tests = {}
    for fold, fam in zip(E.FOLDS, E.FAMILIES):
        t = pd.read_csv(base_root / "manifests" / E.SPLIT_SCHEME / f"fold{fold}" / "test.csv",
                        dtype={"frame": str})
        tests[fam] = set(zip(t["scene"], t["frame"]))
    pool = set().union(*tests.values())
    out = []
    stats = {}
    for tag, r in _variants():
        p = E.radius_variant_root(base_root, tag) / "label_stats.csv"
        if p.exists():
            stats[tag] = pd.read_csv(p, dtype={"frame": str})
    if not stats:
        return out
    any_st = next(iter(stats.values()))
    for name, keys in [*tests.items(), ("all LOSO frames", pool)]:
        row = {"subset": name}
        sel = lambda df: df[[k in keys for k in zip(df["scene"], df["frame"])]]
        b = sel(any_st)
        row["frames"] = len(b)
        allpx = b["n_safe"].sum() + b["n_unsafe"].sum() + b["n_ignore"].sum()
        row[f"safe_{BASE_TAG}"] = b["src_n_safe"].sum() / (b["src_n_safe"].sum() + b["src_n_unsafe"].sum())
        # footprint-only variants were built before src_n_ignore was recorded; their Ignore
        # region is the source's by construction (valid = source slz != 255)
        src_ign = b["src_n_ignore"] if "src_n_ignore" in b else b["n_ignore"]
        row[f"ign_{BASE_TAG}"] = src_ign.sum() / allpx
        row[f"nosafe_{BASE_TAG}"] = int((b["src_n_safe"] == 0).sum())
        for tag, df in stats.items():
            s = sel(df)
            lab = s["n_safe"].sum() + s["n_unsafe"].sum()
            row[f"safe_{tag}"] = s["n_safe"].sum() / lab
            row[f"ign_{tag}"] = s["n_ignore"].sum() / allpx
            row[f"changed_{tag}"] = s["n_changed"].sum() / allpx
            row[f"nosafe_{tag}"] = int((s["n_safe"] == 0).sum())
        out.append(row)
    return out


def _cell(st, nd=3):
    m, sd, n = st
    s = _f(m, nd) + (f" ({_f(sd, nd)})" if sd is not None else "")
    return s + (f" ({n}/{N_FOLDS} folds)" if n < N_FOLDS else "")


def write(out: Path, rows, missing, seed_sd, dl, ls, metric):
    out.mkdir(parents=True, exist_ok=True)
    idx = _index(rows)
    radii = _radii()
    models = E.BASE_MODELS
    lbl = E.label

    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "radius_m", "fold", "seed", *[k for k, _ in METRICS], "json"])
        for r in rows:
            w.writerow([lbl(r["model"]), r["radius"], r["fold"], r["seed"],
                        *[_f(r["metrics"][k]) for k, _ in METRICS], r["json"]])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "radius_m", "n_folds",
                    *[f"{k}_{s}" for k, _ in METRICS for s in ("mean", "cross_scene_sd")]])
        for m in models:
            for tag, r in radii:
                sm = summary(idx, m, tag)
                w.writerow([lbl(m), r, sm["miou"][2], *[_f(v) for k, _ in METRICS for v in sm[k][:2]]])
    with open(out / "per_fold.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "fold", *[f"{k}_r{r:.2f}" for k, _ in METRICS for _, r in radii],
                    *[f"{k}_base_seed_sd" for k, _ in METRICS]])
        for m in models:
            for f in E.FOLDS:
                w.writerow([lbl(m), f, *[_f(idx.get((m, t, f), {}).get(k)) for k, _ in METRICS for t, _ in radii],
                            *[_f(seed_sd[(m, f)][k]) for k, _ in METRICS]])
    with open(out / "delta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "radius_m", "metric", "folds", "d_per_fold (r - base)", "mean_d", "se",
                    "folds_up", "n_folds", "base_mean_seed_sd"])
        for p in dl:
            w.writerow([p["label"], p["radius"], metric, " ".join(map(str, p["folds"])),
                        " ".join(_f(x) for x in p["d"]), _f(p["mean_d"]), _f(p["se"]),
                        p["n_up"], p["n"], _f(p["base_seed_sd"])])
    if ls:
        with open(out / "label_stats.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ls[0]))
            w.writeheader()
            w.writerows(ls)

    rs = ", ".join(f"{r:.2f}/{_pca(t):.2f}" for t, r in radii)
    coupled = any(abs(_pca(t) - E.RADIUS_BASE_PCA_M) > 1e-9 for t, _ in radii)
    pca_rule = ("r_PCA = r + safety buffer 0.15 m" if coupled
                else f"r_PCA fixed at {E.RADIUS_BASE_PCA_M:.2f} m, only the footprint changes")
    L = ["# UAV footprint-radius study", "",
         f"Generated by `tools/summarize_radius.py`. UAV radius r / local-PCA radius r_PCA = {rs} m "
         f"(base {E.RADIUS_BASE_M:.2f}/{E.RADIUS_BASE_PCA_M:.2f} m; {pca_rule}). "
         f"Safety buffer, slope/roughness thresholds, semantic policy, splits and training are unchanged; one seed "
         f"({E.RADIUS_SEED}) per radius, base-radius numbers are the seed-{E.RADIUS_SEED} runs of the main "
         f"experiment. Each run is trained and scored on its own radius' labels, so the TEST targets "
         f"differ between radii: compare columns as 'can this label set be learned and predicted', "
         f"not as a ranking. Fold = experimental unit (n = 4 held-out scenes); sample std (ddof = 1).", ""]
    if ls:
        L += ["## Label statistics", "",
              "Columns are labelled r/r_PCA. Safe = Safe / (Safe + Unsafe) labeled pixels (255 excluded). "
              "Ignore = 255 pixels / all pixels (it changes only with r_PCA: PCA support). Changed = pixels "
              "whose label (0/1/255) differs from the base, / all pixels. No-Safe = frames without a "
              "single Safe pixel. Per held-out family = that fold's test frames.", "",
              "| Test family | frames | " + " | ".join(f"Safe {_rl(t, r)}" for t, r in radii)
              + " | " + " | ".join(f"Ignore {_rl(t, r)}" for t, r in radii)
              + " | " + " | ".join(f"Changed {_rl(t, r)}" for t, r in _variants())
              + " | " + " | ".join(f"No-Safe {_rl(t, r)}" for t, r in radii) + " |",
              "|---|---:|" + "---:|" * (3 * len(radii) + len(_variants()))]
        for row in ls:
            L.append(f"| {row['subset']} | {row['frames']} | "
                     + " | ".join(f"{100 * row[f'safe_{t}']:.2f}%" for t, _ in radii) + " | "
                     + " | ".join(f"{100 * row[f'ign_{t}']:.2f}%" for t, _ in radii) + " | "
                     + " | ".join(f"{100 * row[f'changed_{t}']:.2f}%" for t, _ in _variants()) + " | "
                     + " | ".join(str(row[f"nosafe_{t}"]) for t, _ in radii) + " |")
        L.append("")
    head = "| Model | r / r_PCA (m) | " + " | ".join(l for _, l in METRICS) + " |"
    L += ["## Main table", "", "Cell: mean over the 4 folds (cross-scene std = std of the fold values).", "",
          head, "|---|---:|" + "---:|" * len(METRICS)]
    for m in models:
        for tag, r in radii:
            sm = summary(idx, m, tag)
            if sm["miou"][2]:
                L.append(f"| {lbl(m)} | {r:.2f} / {_pca(tag):.2f} | " + " | ".join(_cell(sm[k]) for k, _ in METRICS) + " |")
    L.append("")
    L += ["## Per fold", "",
          f"mIoU and F1 Safe per radius; 'seed std' = std over the base radius' {len(E.SEEDS)} seeds "
          f"of that fold (the run-to-run noise at a fixed label set).", "",
          "| Model | Fold | " + " | ".join(f"mIoU {_rl(t, r)}" for t, r in radii) + " | mIoU seed std | "
          + " | ".join(f"F1 Safe {_rl(t, r)}" for t, r in radii) + " | F1 seed std |",
          "|---|---:|" + "---:|" * (2 * len(radii) + 2)]
    for m in models:
        for f in E.FOLDS:
            g = lambda t, k: _f(idx.get((m, t, f), {}).get(k), 3) or "–"
            L.append(f"| {lbl(m)} | {f} | " + " | ".join(g(t, "miou") for t, _ in radii)
                     + f" | {_f(seed_sd[(m, f)]['miou'], 3)} | "
                     + " | ".join(g(t, "f1_safe") for t, _ in radii)
                     + f" | {_f(seed_sd[(m, f)]['f1_safe'], 3)} |")
    L.append("")
    name = dict(METRICS)[metric]
    if dl:
        L += [f"## Difference to the base radius ({name})", "",
              f"d_f = value at r − value at the base r = {E.RADIUS_BASE_M:.2f} (same seed, same fold). SE = std(d)/√n. "
              "Up = folds with d > 0. Base seed std = mean over folds of the base radius' seed std: "
              "|d| of that size is within run-to-run noise.", "",
              "| Model | r / r_PCA (m) | d per fold | mean(d) ± SE | up | base seed std |",
              "|---|---:|---|---:|---:|---:|"]
        for p in dl:
            L.append(f"| {p['label']} | {p['radius']:.2f} / {_pca(p['tag']):.2f} | " + " ".join(_f(x, 3) for x in p["d"]) + " | "
                     + _f(p["mean_d"], 3) + (f" ± {_f(p['se'], 3)}" if p["se"] is not None else "")
                     + f" | {p['n_up']}/{p['n']} | {_f(p['base_seed_sd'], 3)} |")
        L.append("")
    L += [f"## Missing ({len(missing)} runs without test_metrics.json)", ""]
    if missing:
        by = {}
        for t in missing:
            by.setdefault(f"{lbl(t['model'])} r={t['radius']:.2f}", []).append(f"f{t['fold']}")
        L += [f"- {k}: {', '.join(v)}" for k, v in by.items()]
    else:
        L.append("- none")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--exp-dir", type=Path, default=None,
                    help="experiment directory (default: the latest outputs/experiments_*)")
    ap.add_argument("--base-root", type=Path, default=None,
                    help="base prepared root (variants next to it) -> label-statistics table")
    ap.add_argument("--out", type=Path, default=None, help="default: <exp-dir>/results_radius")
    ap.add_argument("--metric", default="miou", choices=[k for k, _ in METRICS])
    args = ap.parse_args()
    exp_dir = args.exp_dir or E.latest_experiment_dir()
    if exp_dir is None:
        ap.error("no outputs/experiments_* directory; pass --exp-dir")
    out = args.out or exp_dir / "results_radius"

    rows, missing, seed_sd = collect(exp_dir)
    dl = deltas(_index(rows), seed_sd, args.metric)
    ls = label_stats(args.base_root) if args.base_root else []
    write(out, rows, missing, seed_sd, dl, ls, args.metric)
    print(f"scored: {len(rows)} | missing: {len(missing)} | written: {out}/summary.md")


if __name__ == "__main__":
    main()
