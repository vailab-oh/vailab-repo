# HPO implementation assumptions preserved

This public version uses the final locally evaluated timing revision. Algorithm source and runner mappings are in file_mapping.json.

The original local notes describe a paper-based implementation of Guo et al., DOI 10.1007/s40747-023-01115-2, Algorithms 1-5. No author repository or upstream code license is established.

- Static 2D circular obstacles replace the paper's dynamic 3D perception/prediction setting. Climb limits are inapplicable.
- APF defaults: k_att=20, k_rep=30, rho_0=3, k_bias=5; horizon N_c=10, N_l=15; k_l=1, k_opt=1, epsilon=0.1; turn limit 60 degrees.
- Benchmark-derived step=3, speed=25, goal tolerance=5, neighbor cap/gamma=35, draw budget=6000. delta_t=step/speed=0.12. Do not pass hpo_parameters.json automatically: the existing runner uses it only when --hpo_config is explicitly supplied.
- 256 random draws per planning cycle and 10 executed edges per update are finite-budget implementation assumptions. Rejection samples count toward the same draw budget.
- Parent order defaults to cost, although the paper's textual timestamp ordering is ambiguous. Repulsion uses circle surface distance and summed obstacle forces. Empty frontier continues expansion.
- APF affects new samples. Parent selection does not rewire existing tree nodes. Rerooting preserves valid descendants of the executed endpoint; it is not RRT* rewiring.
- Near-parent edge lengths can exceed one sampling step. The depth-based tree clock is separate from final flight time, which is path length/speed.
- Collision checking is lazy beyond the execution horizon. Goal paths, executed prefixes and returned paths are safety checked.
- The optimizer uses simultaneous-point sweeps, validates combined edges/turns and retains the previous path when improvement is insufficient. No added spline, shortcut or window repair.
- Initial heading is free; subsequent turns and execution boundaries obey the HPO turn check.
- Termination remains first collision-free geometric goal path followed by optimization. No time-window feedback, delay, speed adjustment or loiter is added.
- Evaluation reuses TWA's finished-path evaluator without calling TWA search; the adapter also imports ST's analytic circle collision test. Endpoint/tolerance conventions can differ among baselines.

This benchmark implementation does not claim bitwise equivalence to the original authors' implementation or transfer asymptotic guarantees to the finite-budget adaptation.
