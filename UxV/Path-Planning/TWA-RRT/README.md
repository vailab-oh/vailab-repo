# TWA-RRT*

Research code for **Time-Window-Aware Sampling-Based Path Planning for Time-Constrained Unmanned Aerial Vehicle Missions**, by **Taegeun Oh**, Dong Seoul University. The paper has been accepted.

> **Publication status: Accepted.** Journal, DOI, and final publication details will be updated when available.

TWA-RRT* plans collision-free paths with prescribed arrival-time windows at a fixed flight speed. It combines time-window-aware parent selection and rewiring, late-infeasible pruning, and early-arrival loitering. This repository preserves the experiment implementation and native evaluation rules.

**Release status:** runnable preparation, with unresolved baseline redistribution rights and manuscript/result inconsistencies. Read [the manuscript cross-check](docs/MANUSCRIPT_CROSSCHECK.md) before treating the examples as an exact reconstruction of Table 1 or Fig. 6-8. Baselines are benchmark adaptations, not verified original-author implementations.

## Layout

```text
TWA-RRT/
├── README.md, LICENSE, CITATION.cff, requirements.txt, .gitignore
├── twa_rrt.py
├── run_early_arrival_experiment.py
├── baselines/
│   ├── tb_rrt.py, st_rrt.py, stl_rrt.py
│   └── hpo_rrt/
│       ├── hpo_rrt_star.py, run_hpo_benchmark.py
│       ├── benchmark_adapter.py, hpo_parameters.json
│       └── LICENSE_STATUS.md
├── benchmarks/
│   ├── benchmark_cases_2.json
│   ├── benchmark_cases_M1.json
│   ├── benchmark_cases_N1.json
│   └── benchmark_cases_M1_with_early_arrival.json
├── analysis/generate_comparison_table.py
├── docs/                         # provenance, field audit, checks and limitations
├── example_results/
│   ├── table1/                   # small historical aggregates, not raw runs
│   └── figures/                  # Fig. 5 panel examples from historical trials
└── output/.gitkeep               # generated results; ignored by Git
```

## Installation

Clone the repository and enter the TWA-RRT* project directory before installing
requirements or running experiments:

```bash
git clone https://github.com/vailab-oh/vailab-repo.git
cd vailab-repo/UxV/Path-Planning/TWA-RRT
```

Use Python 3.10 or later; this preparation was execution-tested with Python 3.14.3, NumPy 2.5.3, pandas 3.0.5 and Matplotlib 3.11.2. Historical early-arrival records used Python 3.13.5, NumPy 2.1.3, pandas 2.2.3 and Matplotlib 3.10.0. These are environment records, not a claim that timings are portable. Requirements list only direct third-party imports; SciPy is not used.

```bash
python -m venv .venv
# Activate with .venv/Scripts/Activate.ps1 (PowerShell), or source .venv/bin/activate (POSIX).
python -m pip install -r requirements.txt
```

Run the following commands from this project directory. `--cases`, `--case_id`, `--out_dir`, and `--no_plots` were checked against the actual CLIs. Relative benchmark output paths now resolve inside this repository. Omit `--no_plots` to generate 2D trajectory figures. The public planners do not create 3D maps or x-y-time plots; numeric CSV/JSON outputs are unchanged.

## Benchmarks and protocol

Each of the four input files now contains only its manuscript case (four cases total). The case IDs are moderate, narrow_passage, cluttered and early_arrival. Other cases were removed at the author's request; retained experiment values are unchanged. [Original-to-public mapping](docs/BENCHMARK_SELECTION.json) records the selection. [Case inventory](docs/BENCHMARK_CASES.md) lists every start/goal, obstacle count, time window, reference time, trial count and seed range. [Field audit](docs/BENCHMARK_FIELD_AUDIT.md) explains the single removed legacy parameter and the fields retained for defaults/compatibility. [JSON change log](docs/benchmark_changes.json) records each affected case.

The three main configurations are:

| Geometry | JSON | Case ID | Window (s) | Trials |
|---|---|---|---|---:|
| Moderate, 3 circles | benchmark_cases_2.json | moderate | [7.1, 7.9] | 100 |
| Narrow passage | benchmark_cases_N1.json | narrow_passage | [6.5, 7.3] | 100 |
| Cluttered, 15 circles | benchmark_cases_M1.json | cluttered | [7.5, 8.3] | 100 |
| Cluttered early arrival | benchmark_cases_M1_with_early_arrival.json | early_arrival | [8.6, 9.4] | 5 in input; 100 in the preserved experiment runner |

Names follow Section IV.A and the actual geometry; they do not silently repair the manuscript table labels. The manuscript's Table 1 currently associates its Moderate/Narrow-passage rows with the opposite source cases; see the cross-check.

Main settings: constant speed 25, maximum 6000 iterations/draws, step 3, neighbor radius 35, common goal sampling rate 0.15 (HPO does not use that common bias), seed_base=42 and seed=42+trial-1. A 100-trial run uses seeds 42-141. The retained early-arrival input intentionally keeps 5 trials; its preserved experiment runner produces a separate 100-trial copy. TWA parameters include eps_state=5, w_tw=0.05, w_tw_pred=0.5, delta_late=10, r_hold=20 and fixed time-window mode. Internal `wiser_rrt` keys/classes and `hpo_rrt_r1` output IDs remain for compatibility.

## Run TWA and baselines

Each command below runs the configured 100 trials for the cluttered case, not a smoke test:

```bash
python twa_rrt.py --cases benchmarks/benchmark_cases_M1.json --case_id cluttered
python baselines/tb_rrt.py --cases benchmarks/benchmark_cases_M1.json --case_id cluttered
python baselines/st_rrt.py --cases benchmarks/benchmark_cases_M1.json --case_id cluttered
python baselines/stl_rrt.py --cases benchmarks/benchmark_cases_M1.json --case_id cluttered
python baselines/hpo_rrt/run_hpo_benchmark.py --cases benchmarks/benchmark_cases_M1.json --case_id cluttered
```

Repeat with the JSON and ID in the table above for the other two primary cases. HPO requires both `benchmark_adapter.py` and the TWA/ST modules. Its CLI with no case arguments selects the same three primary cases. `--hpo_config baselines/hpo_rrt/hpo_parameters.json` is optional and is not used by the recorded default runs; explicit overrides are stored with results.

For the original early-arrival 100-trial protocol (all five planners, sequentially):

```bash
python run_early_arrival_experiment.py
```

This creates a separate 100-trial input copy and outputs under `output/early_arrival_experiments/M5_100_<timestamp>/`. The checked-in 5-trial JSON is never changed. Inspect the batch manifest for completion and 100 seeds per method.

For an isolated HPO one-trial smoke check with the unchanged 6000-draw budget:

```bash
python baselines/hpo_rrt/run_hpo_benchmark.py --cases benchmarks/benchmark_cases_N1.json --case_id narrow_passage --smoke --no_plots
```

For other planners, make a temporary JSON copy under `output/`, select one case and set only that copy's num_trials to 1; pass its path via `--cases`. There is no undocumented `--trials` or `--iterations` override.

## Table 1 and Fig. 5

After completing the three primary cases:

```bash
python analysis/generate_comparison_table.py --case "moderate narrow_passage cluttered" --root output --out_root output/analysis --save_raw
```

For the early-arrival batch, substitute the actual batch directory:

```bash
python analysis/generate_comparison_table.py --case early_arrival --root output/early_arrival_experiments/M5_100_<timestamp>/results --out_root output/analysis_early --save_raw
```

The default selects the latest run. `--run_mode all` pools runs and changes the sample set; do not use it for a single 100-trial paper experiment. `selected_runs.json` records what was selected. The preserved overrides `--wiser_run`, `--tb_run`, `--st_run`, `--stl_run`, `--hpo_run` can pin exact runs. Historical run IDs are in example_results/table1/selected_runs.json; full raw result directories are deliberately not bundled.

`comparison_core_metrics_all_cases.csv` provides success rate, successful-trial time error/path length, both all-trial and successful-trial runtime, and several violation definitions. Preserve these distinct columns. Table 1 requires author decisions about the identified label/metric discrepancies; this preparation does not silently rewrite them. HPO completeness checks use summary.csv, raw_trials.jsonl, aggregate.json and config_used.json, all generated by the included runner.

Fig. 5 is the cluttered case's **success-only** `boxplot_time_err_success.png` and `boxplot_runtime_s_success_plain.png`. Keep extra plots enabled (default). Do not use the `time_err_all` plot for the manuscript's successful-trial caption. Assemble the two generated images as panels (a) and (b). The example panels were regenerated from the recorded historical 100-trial pool, not from smoke tests; exact manuscript page layout is not supplied.

## Fig. 6-8

Trajectory plotting lives in each planner; the HPO runner uses the shared TWA plot through its adapter. Run the appropriate case without `--no_plots`, then select the recorded trial images in panel order HPO, ST, STL, TB, TWA:

- Fig. 6: cluttered.
- Fig. 7: narrow_passage (geometry matches the figure, despite the Table 1 label inconsistency).
- Fig. 8: early_arrival from the 100-trial early-arrival batch; the STL panel illustrates a time-window failure.

No separate final Fig. 6-8 composition script, exact trial-selection manifest, or verified original panel asset set was found in the experiment folder. The PDF shows assembled multi-panel pages, but whether the author assembled them manually or with an external script is unverified. To reconstruct a new representative figure, retain the seed/trial selection, copy the corresponding 2D plots, arrange the five panels and add labels/caption. Exact original figures remain unreproducible until those selections/assets are supplied. Existing concept scripts for other figures are not substituted for Fig. 6-8.

## Implementation notes

[Baseline provenance](docs/BASELINE_NOTES.md), [HPO assumptions](docs/HPO_IMPLEMENTATION_NOTES.md), and [measurement protocol](docs/HPO_MEASUREMENT_NOTES.md) detail the adaptations. ST uses a single tree and static obstacles instead of the original bidirectional dynamic setting; STL uses the existing static benchmark cost/runner; TB adds star-style parent selection and rewiring to the cited TB-RRT concept; HPO preserves native geometric-goal termination and a static 2D adaptation. No algorithm or success definition was changed during packaging.

Five paired one-trial checks at seed 42 and 6000 iterations/draws produced identical non-timing CSV metrics between source and public copies. All public planners saved plots, and the comparison reader completed. One STL trial failed the time window in both copies; that is a planning outcome, not a test failure. See [validation](docs/VALIDATION.md). A one-trial check does not certify all 100-trial statistics or exact wall-clock timing.

## Citation and license

Cite Taegeun Oh, *Time-Window-Aware Sampling-Based Path Planning for Time-Constrained Unmanned Aerial Vehicle Missions*, accepted paper. CITATION.cff records the accepted status; journal, DOI, and final publication details will be updated when available.

`LICENSE` grants MIT for **twa_rrt.py only**, as authorized by its author. Redistribution terms for the baseline/analysis/data artifacts remain unconfirmed. Do not present the whole repository as MIT until ownership and any upstream licenses are resolved. The manuscript itself is not bundled.
