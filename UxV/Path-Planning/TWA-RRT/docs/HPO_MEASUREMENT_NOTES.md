# Measurement protocol

The selected HPO revision retains its native search and termination and measures `runtime_s` from immediately before plan() through completion of the external evaluator. Setup, plotting and export are excluded. It separately records setup_s, planning_call_s, evaluation_s, optimizer_s, runtime_with_setup_s, geometric-path events and validated-return events.

`first_sol_time` is geometric-path return time; it is not the first observed time-window-feasible solution. `first_tw_success_observed_s` observes only the final returned result. No unsuccessful time-window trial is promoted to success.

HPO's 6000-draw upper bound does not imply the same actual search work as the other methods. Historical runs and the early-arrival run used different recorded environments. Repeated wall-clock timings are not deterministic and should not be described as controlled algorithmic speedups. Geometry, seed policy, budgets and native metrics are preserved here.

The historical HPO audit reported unchanged paths/iterations for the preceding 300-trial revision comparison; that is historical evidence, not a new 300-trial test performed in this preparation.
