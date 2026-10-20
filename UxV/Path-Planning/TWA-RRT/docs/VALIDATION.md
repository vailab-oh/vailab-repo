# Validation performed

> Subsequent plotting update: TB/ST/STL 3D output blocks were removed; TWA/HPO retain 2D output. Earlier core/plot AST equality describes the initial preparation, not the intentionally changed plotting routines.

> Subsequent selection update: the public inputs now have four cases total, with descriptive IDs. The 21-case checks below describe the initial preparation. See BENCHMARK_SELECTION.json and benchmark_selection_validation.json for the current selection and validation.

- All 9 distributable Python files parsed successfully. Eight runnable/importable modules passed import/CLI checks; the early-arrival orchestration module was imported without starting its 500-trial run, and all script dependencies exist.
- Four benchmark JSONs / 21 case entries passed Draft 2020-12 JSON Schema validation, unique case-ID checks, ordered bound/window checks and v_max=25 checks. Original world/start/goal/obstacles/time/dynamics/random/planner values were compared exactly. Some legacy auxiliary dynamics fields differ among case entries and were preserved.
- Original and public copies were each run once per planner on caseN2_phase2_strict, seed 42, full 6000 iteration/draw budget. Temporary copies alone used num_trials=1. Public runs included plot saving. Original execution used -B and public-only output/cache paths.
- Every non-timing summary.csv field compared equal after excluding only algorithm display labels and timing columns. TWA/TB/ST/HPO achieved time-window success; STL returned success=0 in both copies. All processes exited 0.
- All 112 inspected top-level core algorithm/evaluator/plot/config definitions matched structurally after removing docstrings and normalizing the authorized WISER -> TWA display substitution. Runner/main/normalizer I/O changes were inspected separately; AST comparison does not claim unchanged file/output locations.
- Public comparison script read all five smoke result formats and generated aggregate tables/plots, exit 0. It also reaggregated the historical completed early-arrival 100-trial results, exit 0.
- Fig. 5 success-only example panels were generated from historical M3 pooled trials and visually checked. Exact manuscript panel selection/layout was not asserted.
- 17 selected original code/config/reference files matched their pre-copy SHA-256 hashes. All write operations targeted the new public directory; no source files were moved or renamed. This hash audit is scoped to those 17 files, not a claim of full raw-result-directory hashing.
- No full 100-trial replay was performed. Timings depend on machine, environment and plot/export boundaries and were not compared for equality. One seed is not exhaustive evidence for every possible configuration.

Detailed results: validation.json. Original hashes: source_hashes.json. Final copy mapping/hashes: file_mapping.json. Core code changes: PUBLIC_CHANGES.patch.

## Current selection and 2D-only checks

Four single-case JSONs pass schema, unchanged numeric-condition checks, all four planner normalizers, HPO default selectors and analysis time-window lookup. TB/ST/STL success and failure plotting each saved only a 2D image; their 2D plotting statements match the original. A renamed early_arrival HPO smoke trial saved the shared TWA 2D plot and all CSV/JSON result files, with no 3D image. All 9 Python files parse; audited source hashes are unchanged. Details: benchmark_selection_validation.json. Temporary test output was removed.
