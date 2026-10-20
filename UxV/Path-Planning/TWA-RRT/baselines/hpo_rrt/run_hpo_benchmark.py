"""Standalone HPO-RRT* runner. Existing benchmark algorithms are never run."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parents[1] / 'output' / '.mpl-cache'))
import numpy as np
from benchmark_adapter import ROOT, configure, load_cases, StaticEnvironment, evaluate, smoothness, plot, manuscript
from hpo_rrt_star import HPOConfig, HPORRTStar

# commands.txt's comparison command identifies these representative manuscript
# cases. References only, never duplicate geometry or experiment configuration.
MANUSCRIPT_CASES = [('benchmarks/benchmark_cases_2.json', 'moderate'),
                    ('benchmarks/benchmark_cases_N1.json', 'narrow_passage'),
                    ('benchmarks/benchmark_cases_M1.json', 'cluttered')]


def make_hpo_config(cfg, overrides=None):
    params = asdict(HPOConfig())
    params.update(step_size=cfg.step_size, speed=cfg.v_max,
                  goal_tolerance=cfg.goal_xy_tol, gamma=cfg.rrt_rewire_radius,
                  neighbor_cap=cfg.rrt_rewire_radius, max_iters=cfg.max_iters)
    protected = {'step_size','speed','goal_tolerance','neighbor_cap','max_iters'}
    overrides = overrides or {}
    if protected.intersection(overrides):
        raise ValueError('Benchmark fields cannot be overridden: '+str(protected.intersection(overrides)))
    unknown = set(overrides)-set(params)
    if unknown:
        raise ValueError(f'Unknown HPO parameters: {unknown}')
    params.update(overrides)
    return HPOConfig(**params)


def selected_cases(args):
    if args.cases:
        source = Path(args.cases)
        if not source.is_absolute():
            source = ROOT / source
        result = [(source, c) for c in load_cases(source)
                  if not args.case_id or c['case_id'] == args.case_id]
    else:
        result = []
        for filename, case_id in MANUSCRIPT_CASES:
            if args.case_id and args.case_id != case_id:
                continue
            source = ROOT / filename
            result.extend((source,c) for c in load_cases(source) if c['case_id']==case_id)
    if not result:
        raise ValueError('No matching scenario; specify --cases and --case_id')
    return result


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run_case(source, case, args, overrides):
    cfg, meta = configure(case)
    hpocfg = make_hpo_config(cfg, overrides)
    if args.out_dir:
        out = Path(args.out_dir).resolve()
    else:
        out = Path(case.get('outputs', {}).get('out_dir', 'output'))
        if not out.is_absolute():
            out = (ROOT / out).resolve()
    # Smoke trials cannot be confused with a manuscript Monte Carlo run.
    if args.smoke:
        out = ROOT / 'output' / 'verification_output'
    folder = out / case['case_id'] / 'hpo_rrt_r1' / ('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    json_write(folder/'config_used.json', {'case':case,'resolved_config':asdict(cfg),
        'hpo':asdict(hpocfg),'delta_t':hpocfg.delta_t,'window_metadata':meta,
        'source_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'smoke_only':args.smoke,'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__, 'algorithm_source':hashlib.sha256((HERE/'hpo_rrt_star.py').read_bytes()).hexdigest(),
        'runner_source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'timing_protocol':'r1: runtime_s = planner.plan + external evaluate; setup/plot/export excluded',
        'termination_policy':'first collision-free geometric goal path, then optimizer; no time-window feedback'})
    records = []
    trials = 1 if args.smoke else cfg.num_trials
    with (folder/'raw_trials.jsonl').open('w',encoding='utf-8') as raw, \
         (folder/'summary.csv').open('w',newline='',encoding='utf-8') as summary:
        timing_columns = ['setup_s', 'planning_call_s', 'evaluation_s', 'optimizer_s',
                          'runtime_with_setup_s', 'first_geometric_path_s',
                          'first_geometric_path_iteration', 'validated_return_s',
                          'first_tw_success_observed_s', 'iterations', 'samples', 'created_nodes',
                          'reached_goal', 'termination_reason', 'runtime_protocol']
        writer = csv.DictWriter(summary, fieldnames=manuscript.REFERENCE_HEADER+timing_columns, extrasaction='ignore')
        writer.writeheader()
        for index in range(trials):
            seed = cfg.seed+index
            random.seed(seed)
            np.random.seed(seed)
            setup_start = time.perf_counter()
            env = StaticEnvironment(cfg)
            planner = HPORRTStar(cfg.start[:2],cfg.goal[:2],env,hpocfg)
            setup_s = time.perf_counter()-setup_start
            planning_start = time.perf_counter()
            result = planner.plan()
            planning_call_s = time.perf_counter()-planning_start
            evaluation_start = time.perf_counter()
            evaluated = evaluate(result.points,cfg)
            evaluation_s = time.perf_counter()-evaluation_start
            runtime_s = time.perf_counter()-planning_start
            # Legacy first_sol_time means validated geometric completion, not
            # time-window success. Precise distinct events are exported below.
            first = planning_call_s if result.reached_goal else None
            timing = dict(setup_s=setup_s, planning_call_s=planning_call_s,
                          evaluation_s=evaluation_s, optimizer_s=planner.optimizer_s,
                          runtime_with_setup_s=setup_s+runtime_s,
                          first_geometric_path_s=planner.first_geometric_path_s,
                          first_geometric_path_iteration=planner.first_geometric_path_iteration,
                          validated_return_s=planning_call_s,
                          first_tw_success_observed_s=runtime_s if evaluated['success'] else None,
                          iterations=result.iterations, samples=result.samples,
                          created_nodes=result.created_nodes, reached_goal=result.reached_goal,
                          termination_reason=result.reason, runtime_protocol='r1_plan_plus_evaluation')
            row = manuscript.to_reference_row(case['case_id'],index+1,seed,'HPO-RRT* R1',
                evaluated,runtime_s,len(planner.nodes),first,
                evaluated['L'] if first is not None else None,
                evaluated['total_err'] if first is not None else None,smoothness(evaluated))
            row.update(meta)
            row.update(timing)
            row.update({k:evaluated[k] for k in ('p_tw','t_actual','time_err_actual',
                'time_window_violation_actual','p_tw_actual','success_actual','tmin','tmax')})
            row['fail_reason'] = '' if evaluated['success'] else ('time_window' if result.reached_goal else result.reason)
            record = {'scenario':case['case_id'],'trial':index+1,'seed':seed,
                'target_window':list(cfg.t_window),'target_time':cfg.t_ref,
                'planning':asdict(result),'metrics':{k:v for k,v in evaluated.items()
                    if k not in ('segments','nodes','stats')}, 'row':row,
                'instrumentation':planner.stats,'cycles':planner.cycle_history,
                'timing':dict(timing,runtime_s=runtime_s)}
            raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            raw.flush()  # raw data committed before summary/aggregation
            writer.writerow(row)
            summary.flush()
            json_write(folder/f'trial_{index+1:03d}_path.json',result.points)
            if not args.no_plots:
                plot(cfg,planner.nodes,evaluated,index+1,folder)
            records.append(record)
            print(f"{case['case_id']} {index+1}/{trials} seed={seed} "
                  f"goal={result.reached_goal} success={evaluated['success']} "
                  f"L={evaluated['L']:.3f} t={evaluated['t']:.3f} "
                  f"draws={result.samples} runtime={runtime_s:.3f}s",flush=True)
    keys = ['L','t','spatial_err','time_err','total_err','time_window_violation']
    aggregate = {'scenario':case['case_id'],'trials':trials,'smoke_only':args.smoke,
        'success_rate':statistics.mean(int(r['metrics']['success']) for r in records),
        'goal_reached_rate':statistics.mean(int(r['planning']['reached_goal']) for r in records),
        'metrics':{k:{'mean':statistics.mean(r['metrics'][k] for r in records),
                     'std':statistics.stdev(r['metrics'][k] for r in records) if trials>1 else None}
                   for k in keys},
        'runtime_s_mean':statistics.mean(r['timing']['runtime_s'] for r in records),
        'planning_call_s_mean':statistics.mean(r['timing']['planning_call_s'] for r in records),
        'evaluation_s_mean':statistics.mean(r['timing']['evaluation_s'] for r in records),
        'iterations_mean':statistics.mean(r['planning']['iterations'] for r in records)}
    json_write(folder/'aggregate.json',aggregate)
    print(f'Results: {folder}',flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',help='Existing benchmark JSON; default: three cases from commands.txt')
    parser.add_argument('--case_id',help='Select a case (use --cases for other JSON files)')
    parser.add_argument('--out_dir',help='Output root; relative CLI paths resolve from current directory')
    parser.add_argument('--hpo_config',help='JSON overrides of isolated HPO parameters')
    parser.add_argument('--no_plots',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='One trial, unchanged iteration budget; separate verification output')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    overrides = json.loads(Path(args.hpo_config).read_text(encoding='utf-8')) if args.hpo_config else {}
    for source, case in selected_cases(args):
        run_case(source,case,args,overrides)


if __name__ == '__main__':
    main()
