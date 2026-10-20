# JSON field audit

Only algorithms.st_rrt.dt_min was removed. Its only remaining ST dictionary literal is an unused default-case example; local dt_min in rewiring is computed from spatial distance/v_max, not read from JSON. No dynamic iteration/unpacking of the st_rrt parameter dictionary supplies this key to planning.

| Block | Fields retained and consumers |
|---|---|
| world | xlim, ylim: all planners |
| start/goal | x, y: all; optional theta: TWA/HPO; optional t: ST/STL |
| obstacles | type/cx/cy/r validation and geometry; id retained as provenance/plot metadata |
| time | t_ref, t_window: all; fixed/proxy mode controlled by wiser_rrt |
| dynamics | v_min, v_max, a_max, v0: all configuration readers; retained even where flight time currently uses only v_max |
| random | seed_base, num_trials: all; seed_base + trial - 1 |
| planner | n_iters, step_size, neighbor_radius, goal_sample_rate: common overrides; HPO derives protected parameters and ignores common goal bias |
| tb_rrt | eps_state, dt_min; optional eps_time read with default |
| st_rrt | eps_state; optional lambda/t_max_init/t_max_cap/range_factor/batch_size/max_cond_resamples/max_nodes read with defaults |
| stl_rrt | eps_state, dt_min, stl_weight; internal speed derives from dynamics.v_max |
| wiser_rrt | eps_state, w_tw, w_tw_pred, delta_late, r_hold, time_window_mode; all explicitly read |
| outputs | out_dir retained and changed to output; relative roots consistently point inside repository |

TWA also reads optional goal_xy_tol, w_L, p_tw_type, loiter_max_attempts, collision_step, alpha, beta, kappa. They are not injected into JSONs where absent. Internal wiser_rrt keys remain for compatibility.

## Static access evidence

### twa_rrt.py

```text
142: case_override.get('alpha', alpha)
143: case_override.get('beta', beta)
144: case_override.get('kappa', kappa)
145: case_override.get('time_window_mode', tw_mode)
186: case.get('case_id', 'unknown')
205: obs.get('type')
206: obs.get('type')
249: s.get('theta', 0.0)
250: g.get('theta', 0.0)
262: tm.get('t_ref', (cfg.t_window[0] + cfg.t_window[1]) / 2.0)
284: case.get('outputs', {})
284: case.get('outputs', {}).get('out_dir')
289: case['algorithms'].get('wiser_rrt', {})
1237: self.stats.get('iters_total', 0)
1634: res.get('loiter_len', 0.0)
1641: res.get('success', False)
1647: res.get('spatial_err', 0.0)
1648: res.get('time_err', 0.0)
1649: res.get('total_err', 0.0)
1650: res.get('time_window_violation', 0.0)
1688: case.get('case_id', 'unknown')
1699: case.get('outputs', {})
1699: case.get('outputs', {}).get('out_dir')
1747: case.get('algorithms', {})
1747: case.get('algorithms', {}).get('wiser_rrt', {})
1802: res.get('p_tw', 0.0)
1805: res.get('t_actual', 0.0)
1806: res.get('time_err_actual', 0.0)
1807: res.get('time_window_violation_actual', 0.0)
1808: res.get('p_tw_actual', 0.0)
1809: res.get('success_actual', False)
1810: res.get('tmin', 0.0)
1811: res.get('tmax', 0.0)
1871: res.get('success', False)
1902: row.get('p_tw', '')
1946: row.get('arrival_t', '')
1946: row.get('path_len', '')
1955: res.get('success', False)
2006: c.get('case_id')
```

### baselines/tb_rrt.py

```text
93: os.environ.get('GOAL_BIAS', '0.15')
1128: case.get('outputs', {})
1128: case.get('outputs', {}).get('out_dir', 'output')
1146: case['algorithms'].get('tb_rrt', {})
1155: pln.get('step_size', 3.0)
1156: pln.get('neighbor_radius', 35.0)
1157: pln.get('n_iters', 6000)
1161: pln.get('goal_sample_rate', GOAL_BIAS)
1162: pln.get('neighbor_radius', NEIGHBOR_RADIUS_BASE)
1164: algo_params.get('eps_state', 5.0)
1165: algo_params.get('eps_time', 0.5)
1166: algo_params.get('dt_min', 0.1)
1377: c.get('case_id')
```

### baselines/st_rrt.py

```text
944: eff['algorithms'].get('st_rrt', {})
947: s.get('t', 0.0)
950: algo_params.get('eps_state', 5.0)
952: rnd.get('seed_base', SEED)
953: rnd.get('num_trials', TRIALS)
961: pln.get('n_iters', 6000)
962: pln.get('step_size', 3.0)
963: pln.get('neighbor_radius', 35.0)
964: pln.get('goal_sample_rate', 0.15)
967: algo_params.get('lambda', LAMBDA)
968: algo_params.get('eps_state', 5.0)
969: algo_params.get('t_max_init', T_MAX_INIT)
970: algo_params.get('t_max_cap', T_MAX_CAP)
971: algo_params.get('range_factor', RANGE_FACTOR)
972: algo_params.get('batch_size', BATCH_SIZE)
973: algo_params.get('max_cond_resamples', MAX_COND_RESAMPLES)
974: algo_params.get('max_nodes', 10000)
996: eff.get('outputs', {})
996: eff.get('outputs', {}).get('out_dir', 'output')
1194: data.get('cases', [])
1202: data.get('cases', [])
```

### baselines/stl_rrt.py

```text
380: os.environ.get('FORCE_NO_PLOTS')
1410: eff['algorithms'].get('stl_rrt', {})
1413: s.get('t', 0.0)
1415: algo_params.get('eps_state', 5.0)
1417: rnd.get('seed_base', SEED)
1418: rnd.get('num_trials', TRIALS)
1422: tm.get('t_ref', '')
1426: pln.get('n_iters', 6000)
1427: pln.get('step_size', 5.0)
1428: pln.get('neighbor_radius', 45.0)
1429: pln.get('goal_sample_rate', 0.15)
1432: algo_params.get('dt_min', 0.1)
1433: algo_params.get('stl_weight', 0.6)
1434: algo_params.get('eps_state', 5.0)
1474: eff.get('outputs', {})
1474: eff.get('outputs', {}).get('out_dir', 'output')
1578: eff.get('dynamics', {})
1579: dyn_cfg.get('v_min', 0.0)
1580: dyn_cfg.get('v_max', l_speed)
1581: dyn_cfg.get('a_max', 2.0)
1582: dyn_cfg.get('v0', 0.0)
1766: tc_raw.get('t_lo')
1767: tc_raw.get('t_hi')
1798: data.get('cases', [])
1805: data.get('cases', [])
1811: c.get('case_id')
```

### baselines/hpo_rrt/benchmark_adapter.py

```text
50: case.get('algorithms', {})
50: case.get('algorithms', {}).get('wiser_rrt', {})
```
