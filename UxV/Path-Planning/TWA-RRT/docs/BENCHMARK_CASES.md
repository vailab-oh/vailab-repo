# Public manuscript benchmark cases

One case remains in each JSON. Case names follow Section IV.A and geometry, not the inconsistent first two Table 1 groups. Numeric experiment conditions and seed/trial policies are unchanged.

| JSON | Public ID | Source ID | Circles | Window / reference (s) | Trials | Seeds |
|---|---|---|---:|---|---:|---|
| benchmark_cases_2.json | moderate | caseA3 | 3 | [7.1, 7.9] / 7.5 | 100 | 42-141 |
| benchmark_cases_N1.json | narrow_passage | caseN2_phase2_strict | 3 | [6.5, 7.3] / 6.9 | 100 | 42-141 |
| benchmark_cases_M1.json | cluttered | caseM3_cluttered_detour | 15 | [7.5, 8.3] / 7.9 | 100 | 42-141 |
| benchmark_cases_M1_with_early_arrival.json | early_arrival | caseM5_early_arrival_loiter | 15 | [8.6, 9.4] / 9.0 | 5 | 42-46 |

The early_arrival runner selects the renamed case and creates a separate 100-trial copy (seeds 42-141). All other checked-in cases contain 100 trials. Historical example results and original validation records keep their original source IDs for provenance; map them with BENCHMARK_SELECTION.json. New result directories use the public IDs.
