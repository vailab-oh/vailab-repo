# -*- coding: utf-8 -*-
# R7: aggregate HPO-RRT* R1 results with the existing historical baselines.
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '..', 'output', '.mpl-cache'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import sys
import datetime
import argparse
import re
import json

# Plot style defaults for paper figures: smaller overall size, larger readable text.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 17,
    'lines.markersize': 5,
    'boxplot.boxprops.linewidth': 1.2,
    'boxplot.capprops.linewidth': 1.2,
    'boxplot.whiskerprops.linewidth': 1.2,
    'boxplot.medianprops.linewidth': 1.8,
})

# Script to generate comparison table and box plots for RRT algorithms
# Supports multiple runs, flexible directory structures, specific run overrides, and detailed statistical analysis.

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate RRT Algorithm Comparison Tables and Plots")
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_root = os.path.abspath(os.path.join(script_dir, "output"))
    
    # Required arguments
    parser.add_argument("--case", required=True, help="Case ID (e.g., case_0, case01)")
    
    # Root directory (Default: <script_dir>/../output)
    parser.add_argument("--root", default=default_root, help=f"Root directory for results (Default: {default_root})")
    
    # Run handling
    parser.add_argument("--run_mode", choices=['latest', 'all', 'topN'], default='latest', 
                        help="Run selection mode: latest (newest 1), all, or topN (Default: latest)")
    parser.add_argument("--topN", type=int, default=5, help="Number of latest runs to use if run_mode is topN (Default: 5)")
    
    # Output control
    default_out = os.path.join(script_dir, "output", "analysis")
    parser.add_argument("--out_root", default=default_out, help="Root directory for output summaries")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating boxplots")
    parser.add_argument("--save_raw", action="store_true", help="Save combined raw data CSV")
    
    # Metrics
    default_metrics = "runtime_s,path_len,node_count,arrival_t,spatial_err,time_err,total_err,time_window_violation,first_sol_time,smoothness"
    parser.add_argument("--metrics", default=default_metrics, help="Comma-separated list of metrics to include")

    # --- Extra plot options ---
    parser.add_argument("--no_extra_plots", action="store_true", help="Skip generating additional paper-appeal plots")
    parser.add_argument("--ecdf_only_success", action="store_true", help="ECDF runtime: use success-only trials")
    parser.add_argument("--deadlines", default="0.5,1,2,5,10",
                        help="Comma-separated deadline thresholds in seconds (default: 0.5,1,2,5,10)")

    # Algorithm Directory Overrides
    parser.add_argument("--tb_dir", help="Override TB-RRT* directory")
    parser.add_argument("--st_dir", help="Override ST-RRT* directory")
    parser.add_argument("--stl_dir", help="Override STL-RRT* directory")
    parser.add_argument("--wiser_dir", help="Override TWA-RRT* directory")
    parser.add_argument("--hpo_dir", help="Override HPO-RRT* directory")
    
    # Specific Run Overrides (Priority High)
    parser.add_argument("--tb_run", help="Specific TB-RRT* run path (file or dir)")
    parser.add_argument("--st_run", help="Specific ST-RRT* run path (file or dir)")
    parser.add_argument("--stl_run", help="Specific STL-RRT* run path (file or dir)")
    parser.add_argument("--wiser_run", help="Specific TWA-RRT* run path (file or dir)")
    parser.add_argument("--hpo_run", help="Specific completed HPO-RRT* run path (file or dir)")

    return parser.parse_args()

def load_benchmark_case_info(case_id, script_dir):
    """
    Search repository and benchmarks/ JSONs for the requested case ID.
    Returns dict: {'t_ref': ..., 'tmin': ..., 'tmax': ...} or None.
    """
    patterns = [
        "benchmark_cases.json",
        "benchmark_cases_*.json",
        os.path.join("benchmarks", "benchmark_cases*.json")
    ]
    
    found_files = []
    for p in patterns:
        path_pattern = os.path.join(script_dir, p)
        found_files.extend(glob.glob(path_pattern))
    
    # Unique files
    found_files = sorted(list(set(found_files)))
    
    for fpath in found_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cases = data.get('cases', [])
                for c in cases:
                    if c.get('case_id') == case_id:
                        time_info = c.get('time', {})
                        t_ref = time_info.get('t_ref')
                        t_window = time_info.get('t_window', [])
                        
                        res = {'t_ref': t_ref, 'tmin': None, 'tmax': None}
                        if len(t_window) == 2:
                            res['tmin'] = t_window[0]
                            res['tmax'] = t_window[1]
                        return res
        except Exception as e:
            print(f"  Warning: Could not read/parse {fpath}: {e}")
            
    return None

def resolve_algo_paths(args, case_id=None):
    """
    Resolve effective paths for each algorithm.
    Returns dict: { 'tb_rrt': {'dir': ..., 'run_override': ...}, ... }
    """
    # Use provided case_id or fall back to args.case
    case = case_id if case_id else args.case
    
    # Default folder names under <root>/<case>/
    algo_map = {
        'TB-RRT': {'default_sub': 'tb_rrt', 'dir_arg': args.tb_dir, 'run_arg': args.tb_run},
        'ST-RRT': {'default_sub': 'st_rrt', 'dir_arg': args.st_dir, 'run_arg': args.st_run},
        'STL-RRT': {'default_sub': 'stl_rrt', 'dir_arg': args.stl_dir, 'run_arg': args.stl_run},
        'TWA-RRT*': {'default_sub': 'wiser_rrt', 'dir_arg': args.wiser_dir, 'run_arg': args.wiser_run},
        'HPO-RRT': {'default_sub': 'hpo_rrt_r1', 'dir_arg': args.hpo_dir, 'run_arg': args.hpo_run},
    }
    
    resolved = {}
    base_path = os.path.join(args.root, case)
    
    # If default subfolder doesn't exist, try to find a case-insensitive match
    candidates = {}
    if os.path.exists(base_path):
        for d in os.listdir(base_path):
            candidates[d.lower()] = d
            
    for key, info in algo_map.items():
        # Determine Base Dir
        if info['dir_arg']:
            effective_dir = os.path.abspath(info['dir_arg'])
        else:
            default_sub = info['default_sub']
            # Try exact match first
            target_path = os.path.join(base_path, default_sub)
            if not os.path.exists(target_path) and default_sub.lower() in candidates:
                # Use the actual folder name found (e.g. TB_RRT vs tb_rrt)
                effective_dir = os.path.join(base_path, candidates[default_sub.lower()])
            else:
                effective_dir = target_path
            
        resolved[key] = {
            'dir': effective_dir,
            'run_override': os.path.abspath(info['run_arg']) if info['run_arg'] else None
        }
        
    return resolved

def get_timestamp_from_run_dir(path):
    """
    Extract timestamp from folder name 'run_YYYYMMDD_HHMMSS'.
    Returns datetime object or None.
    """
    basename = os.path.basename(path)
    match = re.search(r'run_(\d{8}_\d{6})', basename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None

def find_summary_in_dir(directory):
    """
    Find summary csv in a directory. 
    Priority: summary.csv > summary_tb.csv > summary_stl.csv
    Returns path or None.
    """
    if os.path.exists(os.path.join(directory, 'INCOMPLETE.json')):
        return None
    candidates = [
        "summary.csv", "summary_tb.csv", "summary_stl.csv",
        "summary_tb2.csv", "summary_st2.csv", "summary_stl2.csv", "summary_wiser2.csv"
    ]
    for c in candidates:
        p = os.path.join(directory, c)
        if os.path.exists(p):
            return p
    return None

def discover_one_summary_from_override(path, algo_key, case_id):
    """
    Handle specific run override. Path can be file or dir.
    """
    if not os.path.exists(path):
        print(f"  [Warning] Override path not found: {path}")
        return []
        
    summary_path = None
    run_dir = None
    run_id = "manual_override"
    
    if os.path.isfile(path):
        summary_path = path
        run_dir = os.path.dirname(path)
        # Try to guess run_id from parent folder name
        parent_name = os.path.basename(run_dir)
        if "run_" in parent_name:
            run_id = parent_name
    else:
        # Directory
        # 1. Check direct
        s = find_summary_in_dir(path)
        if s:
            summary_path = s
            run_dir = path
            if "run_" in os.path.basename(path):
                run_id = os.path.basename(path)
        else:
            # 2. Check strict subdirs (depth 1)
            # Find any subfolder with summary.csv?
            # Or assume the user gave the run folder and maybe it's inside 'subdir'? 
            # Let's search recursively depth 2
            found = False
            for root, dirs, files in os.walk(path):
                # Depth check: relative path depth
                rel = os.path.relpath(root, path)
                if rel.count(os.sep) >= 2: continue # Limit depth
                
                s = find_summary_in_dir(root)
                if s:
                    summary_path = s
                    run_dir = root
                    # logic to extract run_id?
                    run_id = os.path.basename(path) # Use the provided override folder name as ID base
                    found = True
                    break
            if not found:
                print(f"  [Warning] No summary.csv found in override dir: {path}")
                return []

    if not run_is_complete(run_dir, algo_key):
        raise ValueError(f'Incomplete or smoke run cannot be selected: {run_dir}')
    return [{
        "algo": algo_key,
        "case_id": case_id,
        "run_dir": run_dir,
        "run_id": run_id,
        "summary_csv": summary_path
    }]

def run_is_complete(directory, algo_key):
    if os.path.exists(os.path.join(directory, 'INCOMPLETE.json')):
        return False
    if algo_key != 'HPO-RRT':
        return True  # Legacy baselines do not have HPO completion metadata.
    try:
        with open(os.path.join(directory, 'aggregate.json'), encoding='utf-8') as f:
            aggregate = json.load(f)
        with open(os.path.join(directory, 'config_used.json'), encoding='utf-8') as f:
            config = json.load(f)
        expected = config['case']['random']['num_trials']
        return (not aggregate.get('smoke_only', False)
                and not config.get('smoke_only', False)
                and aggregate['trials'] == expected
                and len(pd.read_csv(os.path.join(directory, 'summary.csv'))) == expected)
    except (OSError, ValueError, KeyError):
        return False


def enrich_hpo_summary(df, rinfo):
    """Use actual goal flags from raw records, not partial-path evidence."""
    if rinfo['algo'] != 'HPO-RRT':
        return df
    path = os.path.join(rinfo['run_dir'], 'raw_trials.jsonl')
    with open(path, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    lookup = {(r['trial'], r['seed']): r for r in records}
    if len(lookup) != len(records) or len(records) != len(df):
        raise ValueError('HPO raw/CSV trial counts differ or duplicate trials exist')
    result = df.copy()
    matched = [lookup[(int(row.trial), int(row.seed))] for row in result.itertuples()]
    if any(r['scenario'] != rinfo['case_id'] for r in matched):
        raise ValueError('HPO scenario mismatch')
    result['path_found'] = [int(r['planning']['reached_goal']) for r in matched]
    return result


def discover_run_summaries(base_dir, algo_key, case_id, run_mode, topN):
    """
    Scan base_dir for run_* folders.
    Sort by timestamp (run_YYYYMMDD...) -> mtime.
    """
    if not os.path.exists(base_dir):
        # Silent fail or warn? Only warn if it looked like it should exist?
        # Standard warning
        print(f"  [Info] Algo dir not found (skipping): {base_dir}")
        return []
        
    # Pattern: <algo_dir>/run_* 
    # (The requirement says <algo_dir> is likely .../tb_rrt, inside which are run_* folders)
    
    pattern = os.path.join(base_dir, "run_*")
    candidates = glob.glob(pattern)
    valid_runs = []
    
    for c in candidates:
        if os.path.isdir(c) and run_is_complete(c, algo_key) and find_summary_in_dir(c):
            valid_runs.append(c)
            
    if not valid_runs:
         print(f"  [Info] No 'run_*' folders in: {base_dir}")
         return []
         
    # Sort
    def sort_key(p):
        ts = get_timestamp_from_run_dir(p)
        if ts:
            return ts.timestamp()
        return os.path.getmtime(p)
        
    valid_runs.sort(key=sort_key, reverse=True) # Newest first
    
    # Filter
    if run_mode == 'latest':
        selected = valid_runs[:1]
    elif run_mode == 'topN':
        selected = valid_runs[:topN]
    else:
        selected = valid_runs
        
    results = []
    for r_dir in selected:
        s_csv = find_summary_in_dir(r_dir)
        if s_csv:
            results.append({
                "algo": algo_key,
                "case_id": case_id,
                "run_dir": r_dir,
                "run_id": os.path.basename(r_dir),
                "summary_csv": s_csv
            })
            
    return results

def normalize_metrics(df, meta_info):
    """
    Standardize DataFrame columns and content.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    
    # 1. Clean column names and handle duplicates
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # 1. Inject Metadata
    df['algo'] = meta_info['algo']
    df['run_id'] = meta_info['run_id']
    if 'case_id' not in df.columns:
        df['case_id'] = meta_info['case_id']
        
    # Helper to find col
    def get_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Trial
    trial_col = get_col(['trial', 'trial_id', 'iter'])
    if trial_col:
        df['trial'] = df[trial_col]
    else:
        df['trial'] = range(len(df))

    # Success (0/1)
    succ_col = get_col(['success', 'success_flag']) 
    if succ_col:
        df['success'] = _normalize_binary_flag(df[succ_col])
    else:
        df['success'] = np.nan

    # Runtime (s)
    rt_col = get_col(['runtime_s', 'runtime', 'execution_time_s', 'execution_time_ms', 'time_s'])
    if rt_col:
        vals = pd.to_numeric(df[rt_col], errors='coerce')
        # Heuristic: if median > 100, assume ms
        median_val = vals.median()
        is_ms = False
        if '_ms' in rt_col:
            is_ms = True
        elif 'execution_time' in rt_col and '_s' not in rt_col and (pd.notna(median_val) and median_val > 100):
             is_ms = True
             
        if is_ms:
            df['runtime_s'] = vals / 1000.0
        else:
            df['runtime_s'] = vals
    else:
        df['runtime_s'] = np.nan

    # Path Length
    len_col = get_col(['path_len', 'path_length_m', 'length', 'cost_length'])
    df['path_len'] = pd.to_numeric(df[len_col], errors='coerce') if len_col else np.nan

    # Node Count
    node_col = get_col(['node_count', 'nodes', 'tree_size'])
    df['node_count'] = pd.to_numeric(df[node_col], errors='coerce') if node_col else np.nan

    # Arrival Time / Reference Times
    arr_col = get_col(['arrival_t', 'arrival_time', 'arrival'])
    df['arrival_t'] = pd.to_numeric(df[arr_col], errors='coerce') if arr_col else np.nan

    t_ref_col = get_col(['t_ref', 'tref', 'time_ref'])
    df['t_ref'] = pd.to_numeric(df[t_ref_col], errors='coerce') if t_ref_col else np.nan

    # Prefer Tmin/Tmax (likely benchmark) over tmin/tmax (maybe trial-specific)
    tmin_col = get_col(['tmin', 't_min', 't_window_min'])
    df['tmin'] = pd.to_numeric(df[tmin_col], errors='coerce') if tmin_col else np.nan

    tmax_col = get_col(['tmax', 't_max', 't_window_max'])
    df['tmax'] = pd.to_numeric(df[tmax_col], errors='coerce') if tmax_col else np.nan

    # Errors
    sp_col = get_col(['spatial_err', 'spatial_error', 'dist_err'])
    df['spatial_err'] = pd.to_numeric(df[sp_col], errors='coerce') if sp_col else np.nan
    
    tm_col = get_col(['time_err', 'time_error'])
    df['time_err'] = pd.to_numeric(df[tm_col], errors='coerce') if tm_col else np.nan
    
    tot_col = get_col(['total_err', 'total_error'])
    df['total_err'] = pd.to_numeric(df[tot_col], errors='coerce') if tot_col else np.nan

    # Time Window Violation
    tw_col = get_col(['time_window_violation', 'tw_violation', 'violation'])
    df['time_window_violation'] = pd.to_numeric(df[tw_col], errors='coerce') if tw_col else np.nan

    # Optional explicit path-found / goal-reached flag from source summary
    path_col = get_col(['path_found', 'found_path', 'has_path', 'goal_reached', 'reached_goal', 'spatial_success', 'path_success'])
    if path_col:
        df['path_found'] = _normalize_binary_flag(df[path_col])
    else:
        df['path_found'] = np.nan

    # First Solution Time
    fst_col = get_col(['first_sol_time', 'first_path_time', 'first_sol_time_s'])
    if fst_col:
        vals = pd.to_numeric(df[fst_col], errors='coerce')
        if (pd.notna(vals.median()) and vals.median() > 100) and '_s' not in fst_col: 
            df['first_sol_time'] = vals / 1000.0
        else:
             df['first_sol_time'] = vals
    else:
        df['first_sol_time'] = np.nan

    # Smoothness
    sm_col = get_col(['smoothness', 'path_smoothness'])
    df['smoothness'] = pd.to_numeric(df[sm_col], errors='coerce') if sm_col else np.nan

    # Derive final success if source omitted it
    if df['success'].isna().all():
        if 'path_found' in df.columns and pd.to_numeric(df['path_found'], errors='coerce').notna().any():
            tw_ok = df['time_window_violation'].fillna(0).le(1e-12) if 'time_window_violation' in df.columns else True
            df['success'] = ((_normalize_binary_flag(df['path_found']) == 1) & tw_ok).astype(int)
        else:
            df['success'] = 0
    else:
        df['success'] = _normalize_binary_flag(df['success'])

    # Canonical columns
    canonical_cols = [
        'case_id', 'algo', 'run_id', 'trial', 'seed', 'success', 'path_found',
        'runtime_s', 'path_len', 'node_count', 'arrival_t', 
        'spatial_err', 'time_err', 'total_err', 'time_window_violation', 
        'first_sol_time', 'smoothness', 't_ref', 'tmin', 'tmax'
    ]
    
    # Fill missing
    for c in canonical_cols:
        if c not in df.columns:
            df[c] = np.nan
            
    return add_outcome_flags(df[canonical_cols])


def tw_violation_split(series, eps=1e-12):
    """Split time-window violation into (rate %, mean magnitude, max magnitude).

    series: numeric violation values (0 = no violation, >0 = violation magnitude).
    eps: small tolerance to avoid treating tiny numerical noise as violation.
    """
    if series is None:
        return (np.nan, np.nan, np.nan)

    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return (np.nan, np.nan, np.nan)

    viol = s[s > eps]
    rate = (len(viol) / len(s) * 100.0) if len(s) > 0 else np.nan
    if len(viol) == 0:
        return (rate, 0.0, 0.0)

    return (rate, float(viol.mean()), float(viol.max()))


def _normalize_binary_flag(series):
    """Best-effort conversion of mixed binary-like values to 0/1 ints."""
    if series is None:
        return None

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    s = series.copy()
    numeric = pd.to_numeric(s, errors='coerce')
    if numeric.notna().any():
        return numeric.fillna(0).gt(0).astype(int)

    text = s.astype(str).str.strip().str.lower()
    true_tokens = {'1', 'true', 't', 'yes', 'y', 'success', 'found', 'reachable', 'reached'}
    false_tokens = {'0', 'false', 'f', 'no', 'n', 'fail', 'failed', 'none', 'nan'}
    mapped = pd.Series(np.nan, index=s.index, dtype='float64')
    mapped[text.isin(true_tokens)] = 1.0
    mapped[text.isin(false_tokens)] = 0.0
    return mapped.fillna(0).astype(int)


def add_outcome_flags(df, eps=1e-12):
    """Augment normalized trial dataframe with spatial/temporal outcome factors.

    Definitions
    -----------
    - path_found: a spatial path / goal-reaching solution exists.
    - success: final success (assumed to include TW satisfaction if source defines it so).
    - path_found_but_final_fail: path exists, but final success is 0.
    - path_found_but_tw_fail: path exists, final success is 0, and TW violation > 0.
    - no_path: no spatial path evidence.
    - temporal_success_given_path_pct: computed later as success / path_found.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # success should already be normalized to 0/1, but re-guard here.
    out['success'] = _normalize_binary_flag(out['success']) if 'success' in out.columns else 0

    # Path-found inference: explicit flag preferred, otherwise use path evidence columns.
    explicit_path_col = None
    for c in ['path_found', 'goal_reached', 'reached_goal', 'has_path', 'spatial_success', 'path_success']:
        if c in out.columns:
            explicit_path_col = c
            break

    if explicit_path_col is not None and out[explicit_path_col].notna().any():
        path_found = _normalize_binary_flag(out[explicit_path_col])
    else:
        evidence_cols = [
            'path_len', 'arrival_t', 'spatial_err', 'time_err', 'total_err',
            'first_sol_time', 'smoothness', 'time_window_violation'
        ]
        evidence = pd.Series(False, index=out.index)
        for c in evidence_cols:
            if c in out.columns:
                evidence = evidence | pd.to_numeric(out[c], errors='coerce').notna()
        path_found = ((out['success'] == 1) | evidence).astype(int)

    out['path_found'] = path_found.astype(int)

    # Temporal satisfaction: TW violation <= eps means satisfied. If unavailable, fallback to success.
    if 'time_window_violation' in out.columns:
        tw = pd.to_numeric(out['time_window_violation'], errors='coerce')
        tw_satisfied = tw.le(eps)
        tw_satisfied = tw_satisfied.where(tw.notna(), out['success'] == 1)
        out['tw_satisfied'] = tw_satisfied.astype(int)
        out['tw_viol_positive'] = tw.gt(eps).fillna(False).astype(int)
    else:
        out['tw_satisfied'] = (out['success'] == 1).astype(int)
        out['tw_viol_positive'] = 0

    out['path_found_but_final_fail'] = ((out['path_found'] == 1) & (out['success'] == 0)).astype(int)
    out['path_found_but_tw_fail'] = ((out['path_found'] == 1) & (out['success'] == 0) & (out['tw_viol_positive'] == 1)).astype(int)
    out['no_path'] = (out['path_found'] == 0).astype(int)

    outcome = np.where(
        out['success'] == 1,
        'full_success',
        np.where(
            out['path_found_but_tw_fail'] == 1,
            'path_found_but_tw_fail',
            np.where(
                out['path_found_but_final_fail'] == 1,
                'path_found_but_final_fail',
                'no_path'
            )
        )
    )
    out['outcome_category'] = outcome
    return out


def calculate_stats(df, output_root_case):
    """
    Calculate Run-level and Pooled statistics.
    """
    if df.empty:
        return pd.DataFrame()

    perf_metrics = ['runtime_s', 'path_len', 'node_count', 'first_sol_time', 'smoothness', 'arrival_t']
    error_metrics = ['spatial_err', 'time_err', 'total_err', 'time_window_violation']
    
    # --- A) Run-level Statistics ---
    run_stats = []
    groups = df.groupby(['algo', 'run_id'])
    
    for (algo, run_id), group in groups:
        total_trials = len(group)
        succ_group = group[group['success'] == 1]
        success_count = len(succ_group)
        success_rate = (success_count / total_trials * 100.0) if total_trials > 0 else 0.0
        
        path_group = group[group['path_found'] == 1] if 'path_found' in group.columns else pd.DataFrame(columns=group.columns)
        path_count = len(path_group)
        path_found_rate = (path_count / total_trials * 100.0) if total_trials > 0 else 0.0
        path_fail_count = int(((group['path_found'] == 1) & (group['success'] == 0)).sum()) if 'path_found' in group.columns else 0
        path_tw_fail_count = int(((group['path_found'] == 1) & (group['success'] == 0) & (group['tw_viol_positive'] == 1)).sum()) if 'tw_viol_positive' in group.columns else 0
        no_path_count = int((group['path_found'] == 0).sum()) if 'path_found' in group.columns else 0
        temporal_success_given_path = (success_count / path_count * 100.0) if path_count > 0 else np.nan

        row = {
            'algo': algo,
            'run_id': run_id,
            'trials': total_trials,
            'success_rate': success_rate,
            'path_found_rate': path_found_rate,
            'path_found_but_final_fail_rate': (path_fail_count / total_trials * 100.0) if total_trials > 0 else np.nan,
            'path_found_but_tw_fail_rate': (path_tw_fail_count / total_trials * 100.0) if total_trials > 0 else np.nan,
            'no_path_rate': (no_path_count / total_trials * 100.0) if total_trials > 0 else np.nan,
            'temporal_success_given_path': temporal_success_given_path,
        }
        
        # Performance (Success Only)
        for m in perf_metrics:
            if m == 'runtime_s':
                # Split runtime into All and Success
                row['runtime_all_mean'] = group[m].mean()
                row['runtime_all_std'] = group[m].std(ddof=1) if len(group)>1 else 0.0
                if not succ_group.empty:
                    row['runtime_success_mean'] = succ_group[m].mean()
                    row['runtime_success_std'] = succ_group[m].std(ddof=1) if len(succ_group)>1 else 0.0
                else:
                    row['runtime_success_mean'] = np.nan
                    row['runtime_success_std'] = np.nan
            else:
                 # Default success only
                if not succ_group.empty and m in succ_group:
                    mean_val = succ_group[m].mean()
                    std_val = succ_group[m].std(ddof=1) if len(succ_group)>1 else 0.0
                    row[f'{m}_mean'] = mean_val
                    row[f'{m}_std'] = std_val
                else:
                    row[f'{m}_mean'] = np.nan
                    row[f'{m}_std'] = np.nan
        
        # Error (All)
        for m in error_metrics:
            if m in group:
                row[f'{m}_mean'] = group[m].mean()
                row[f'{m}_std'] = group[m].std(ddof=1) if len(group)>1 else 0.0
            else:
                 row[f'{m}_mean'] = np.nan
                 row[f'{m}_std'] = np.nan
                 
        
        # Time-window violation split (rate + magnitude)
        if 'time_window_violation' in group.columns:
            r_all, m_all, mx_all = tw_violation_split(group['time_window_violation'])
            row['tw_viol_rate_all'] = r_all
            row['tw_viol_mean_mag_all'] = m_all
            row['tw_viol_max_mag_all'] = mx_all

            r_s, m_s, mx_s = tw_violation_split(succ_group['time_window_violation']) if not succ_group.empty else (np.nan, np.nan, np.nan)
            row['tw_viol_rate_success'] = r_s
            row['tw_viol_mean_mag_success'] = m_s
            row['tw_viol_max_mag_success'] = mx_s
        else:
            row['tw_viol_rate_all'] = np.nan
            row['tw_viol_mean_mag_all'] = np.nan
            row['tw_viol_max_mag_all'] = np.nan
            row['tw_viol_rate_success'] = np.nan
            row['tw_viol_mean_mag_success'] = np.nan
            row['tw_viol_max_mag_success'] = np.nan

        run_stats.append(row)
        
    run_level_df = pd.DataFrame(run_stats)
    
    # Run-level Aggregation (Mean +/- Std of runs)
    algo_summary_rows = []
    if not run_level_df.empty:
        for algo, grp in run_level_df.groupby('algo'):
            s_row = {'algo': algo, 'runs_count': len(grp)}
            
            # Helper for mean/std of run means
            def add_stat_of_means(metric_col_name, out_prefix):
                if metric_col_name not in grp: 
                    s_row[f'{out_prefix}_mean'] = np.nan
                    s_row[f'{out_prefix}_std'] = np.nan
                    return
                series = grp[metric_col_name].dropna()
                if series.empty:
                    s_row[f'{out_prefix}_mean'] = np.nan
                    s_row[f'{out_prefix}_std'] = np.nan
                else:
                    s_row[f'{out_prefix}_mean'] = series.mean()
                    s_row[f'{out_prefix}_std'] = series.std(ddof=1) if len(series) > 1 else 0.0

            add_stat_of_means('success_rate', 'success_rate')
            add_stat_of_means('path_found_rate', 'path_found_rate')
            add_stat_of_means('path_found_but_final_fail_rate', 'path_found_but_final_fail_rate')
            add_stat_of_means('path_found_but_tw_fail_rate', 'path_found_but_tw_fail_rate')
            add_stat_of_means('no_path_rate', 'no_path_rate')
            add_stat_of_means('temporal_success_given_path', 'temporal_success_given_path')
            add_stat_of_means('runtime_all_mean', 'runtime_all')
            add_stat_of_means('runtime_success_mean', 'runtime_success')
            
            for m in perf_metrics:
                if m == 'runtime_s': continue
                add_stat_of_means(f'{m}_mean', m)
                
            for m in error_metrics:
                add_stat_of_means(f'{m}_mean', m)

            # Time-window violation split (rate + magnitude)
            add_stat_of_means('tw_viol_rate_all', 'tw_viol_rate_all')
            add_stat_of_means('tw_viol_mean_mag_all', 'tw_viol_mean_mag_all')
            add_stat_of_means('tw_viol_max_mag_all', 'tw_viol_max_mag_all')
            add_stat_of_means('tw_viol_rate_success', 'tw_viol_rate_success')
            add_stat_of_means('tw_viol_mean_mag_success', 'tw_viol_mean_mag_success')
            add_stat_of_means('tw_viol_max_mag_success', 'tw_viol_max_mag_success')
            
            algo_summary_rows.append(s_row)
            
    algo_runlevel_summary = pd.DataFrame(algo_summary_rows)
    
    # Save Run-level Details
    run_level_df.to_csv(os.path.join(output_root_case, "comparison_stats_runlevel_details.csv"), index=False)
    
    # Save Aggregated
    algo_runlevel_summary.to_csv(os.path.join(output_root_case, "comparison_stats_runlevel.csv"), index=False)
    
    # --- B) Pooled Statistics ---
    pooled_rows = []
    if not df.empty:
        for algo, grp in df.groupby('algo'):
            succ_grp = grp[grp['success'] == 1]
            total = len(grp)
            sr = (len(succ_grp) / total * 100.0) if total > 0 else 0.0
            
            path_grp = grp[grp['path_found'] == 1] if 'path_found' in grp.columns else pd.DataFrame(columns=grp.columns)
            n_path = len(path_grp)
            n_path_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0)).sum()) if 'path_found' in grp.columns else 0
            n_path_tw_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0) & (grp['tw_viol_positive'] == 1)).sum()) if 'tw_viol_positive' in grp.columns else 0
            n_no_path = int((grp['path_found'] == 0).sum()) if 'path_found' in grp.columns else 0

            p_row = {
                'algo': algo,
                'total_trials': total,
                'success_rate': sr,
                'path_found_rate': (n_path / total * 100.0) if total > 0 else np.nan,
                'path_found_but_final_fail_rate': (n_path_fail / total * 100.0) if total > 0 else np.nan,
                'path_found_but_tw_fail_rate': (n_path_tw_fail / total * 100.0) if total > 0 else np.nan,
                'no_path_rate': (n_no_path / total * 100.0) if total > 0 else np.nan,
                'temporal_success_given_path': (len(succ_grp) / n_path * 100.0) if n_path > 0 else np.nan,
            }
            
            # Runtime All
            p_row['runtime_all_mean'] = grp['runtime_s'].mean()
            p_row['runtime_all_std'] = grp['runtime_s'].std(ddof=1) if total > 1 else 0.0

            # Runtime Success
            if not succ_grp.empty:
                 p_row['runtime_success_mean'] = succ_grp['runtime_s'].mean()
                 p_row['runtime_success_std'] = succ_grp['runtime_s'].std(ddof=1) if len(succ_grp) > 1 else 0.0
            else:
                 p_row['runtime_success_mean'] = np.nan
                 p_row['runtime_success_std'] = np.nan

            for m in perf_metrics:
                if m == 'runtime_s': continue
                if m in succ_grp:
                    p_row[f'{m}_mean'] = succ_grp[m].mean()
                    p_row[f'{m}_std'] = succ_grp[m].std(ddof=1) if len(succ_grp) > 1 else 0.0
                else:
                    p_row[f'{m}_mean'] = np.nan
                    p_row[f'{m}_std'] = np.nan
            
            for m in error_metrics:
                if m in grp:
                    p_row[f'{m}_mean'] = grp[m].mean()
                    p_row[f'{m}_std'] = grp[m].std(ddof=1) if total > 1 else 0.0
                else:
                    p_row[f'{m}_mean'] = np.nan
                    p_row[f'{m}_std'] = np.nan
                
            
            # Time-window violation split (rate + magnitude)
            if 'time_window_violation' in grp.columns:
                r_all, m_all, mx_all = tw_violation_split(grp['time_window_violation'])
                p_row['tw_viol_rate_all'] = r_all
                p_row['tw_viol_mean_mag_all'] = m_all
                p_row['tw_viol_max_mag_all'] = mx_all

                r_s, m_s, mx_s = tw_violation_split(succ_grp['time_window_violation']) if not succ_grp.empty else (np.nan, np.nan, np.nan)
                p_row['tw_viol_rate_success'] = r_s
                p_row['tw_viol_mean_mag_success'] = m_s
                p_row['tw_viol_max_mag_success'] = mx_s
            else:
                p_row['tw_viol_rate_all'] = np.nan
                p_row['tw_viol_mean_mag_all'] = np.nan
                p_row['tw_viol_max_mag_all'] = np.nan
                p_row['tw_viol_rate_success'] = np.nan
                p_row['tw_viol_mean_mag_success'] = np.nan
                p_row['tw_viol_max_mag_success'] = np.nan

            pooled_rows.append(p_row)
            
    pooled_df = pd.DataFrame(pooled_rows)

    pooled_df.to_csv(os.path.join(output_root_case, "comparison_stats_pooled.csv"), index=False)
    
    return run_level_df


def save_core_metrics_table(df, out_dir):
    """Save a compact algorithm-level summary focused on paper-ready core metrics.

    Notes
    -----
    - 'all' means all trials, regardless of success/failure.
    - 'success' means success subset only.
    - time_err/spatial_err/total_err are reported on success trials as primary quality metrics,
      because failures often contain NaN or non-comparable values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for algo, grp in df.groupby('algo'):
        grp = grp.copy()
        succ = grp[grp['success'] == 1]
        total = len(grp)
        n_succ = len(succ)

        tw_rate_all, tw_mean_all, tw_max_all = tw_violation_split(grp['time_window_violation']) if 'time_window_violation' in grp.columns else (np.nan, np.nan, np.nan)
        tw_rate_s, tw_mean_s, tw_max_s = tw_violation_split(succ['time_window_violation']) if (n_succ > 0 and 'time_window_violation' in succ.columns) else (np.nan, np.nan, np.nan)

        path_found = grp[grp['path_found'] == 1] if 'path_found' in grp.columns else pd.DataFrame(columns=grp.columns)
        n_path = len(path_found)
        n_path_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0)).sum()) if 'path_found' in grp.columns else 0
        n_path_tw_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0) & (grp['tw_viol_positive'] == 1)).sum()) if 'tw_viol_positive' in grp.columns else 0
        n_no_path = int((grp['path_found'] == 0).sum()) if 'path_found' in grp.columns else 0
        tw_rate_path, tw_mean_path, tw_max_path = tw_violation_split(path_found['time_window_violation']) if (n_path > 0 and 'time_window_violation' in path_found.columns) else (np.nan, np.nan, np.nan)

        row = {
            'algo': algo,
            'total_trials': int(total),
            'success_trials': int(n_succ),
            'path_found_trials': int(n_path),
            'success_rate_pct': float(n_succ / total * 100.0) if total > 0 else np.nan,
            'path_found_rate_pct': float(n_path / total * 100.0) if total > 0 else np.nan,
            'path_found_but_final_fail_rate_pct': float(n_path_fail / total * 100.0) if total > 0 else np.nan,
            'path_found_but_tw_fail_rate_pct': float(n_path_tw_fail / total * 100.0) if total > 0 else np.nan,
            'no_path_rate_pct': float(n_no_path / total * 100.0) if total > 0 else np.nan,
            'temporal_success_given_path_pct': float(n_succ / n_path * 100.0) if n_path > 0 else np.nan,

            'runtime_all_mean_s': float(pd.to_numeric(grp['runtime_s'], errors='coerce').mean()) if 'runtime_s' in grp.columns else np.nan,
            'runtime_success_mean_s': float(pd.to_numeric(succ['runtime_s'], errors='coerce').mean()) if ('runtime_s' in succ.columns and n_succ > 0) else np.nan,
            
            'path_len_success_mean_m': float(pd.to_numeric(succ['path_len'], errors='coerce').mean()) if ('path_len' in succ.columns and n_succ > 0) else np.nan,

            'arrival_t_success_mean_s': float(pd.to_numeric(succ['arrival_t'], errors='coerce').mean()) if ('arrival_t' in succ.columns and n_succ > 0) else np.nan,
            'spatial_err_success_mean': float(pd.to_numeric(succ['spatial_err'], errors='coerce').mean()) if ('spatial_err' in succ.columns and n_succ > 0) else np.nan,
            'time_err_success_mean_s': float(pd.to_numeric(succ['time_err'], errors='coerce').mean()) if ('time_err' in succ.columns and n_succ > 0) else np.nan,
            'total_err_success_mean': float(pd.to_numeric(succ['total_err'], errors='coerce').mean()) if ('total_err' in succ.columns and n_succ > 0) else np.nan,

            'tw_viol_rate_all_pct': float(tw_rate_all) if pd.notna(tw_rate_all) else np.nan,
            'tw_viol_mean_mag_all_s': float(tw_mean_all) if pd.notna(tw_mean_all) else np.nan,
            'tw_viol_max_mag_all_s': float(tw_max_all) if pd.notna(tw_max_all) else np.nan,
            'tw_viol_rate_path_found_pct': float(tw_rate_path) if pd.notna(tw_rate_path) else np.nan,
            'tw_viol_mean_mag_path_found_s': float(tw_mean_path) if pd.notna(tw_mean_path) else np.nan,
            'tw_viol_max_mag_path_found_s': float(tw_max_path) if pd.notna(tw_max_path) else np.nan,

            'tw_viol_rate_success_pct': float(tw_rate_s) if pd.notna(tw_rate_s) else np.nan,
            'tw_viol_mean_mag_success_s': float(tw_mean_s) if pd.notna(tw_mean_s) else np.nan,
            'tw_viol_max_mag_success_s': float(tw_max_s) if pd.notna(tw_max_s) else np.nan,
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        ordered_cols = [
            'algo', 'total_trials', 'success_trials', 'path_found_trials',
            'success_rate_pct', 'path_found_rate_pct', 'path_found_but_final_fail_rate_pct',
            'path_found_but_tw_fail_rate_pct', 'no_path_rate_pct', 'temporal_success_given_path_pct',
            'runtime_all_mean_s', 'runtime_success_mean_s', 'path_len_success_mean_m',
            'arrival_t_success_mean_s', 'spatial_err_success_mean', 'time_err_success_mean_s', 'total_err_success_mean',
            'tw_viol_rate_all_pct', 'tw_viol_mean_mag_all_s', 'tw_viol_max_mag_all_s',
            'tw_viol_rate_path_found_pct', 'tw_viol_mean_mag_path_found_s', 'tw_viol_max_mag_path_found_s',
            'tw_viol_rate_success_pct', 'tw_viol_mean_mag_success_s', 'tw_viol_max_mag_success_s',
        ]
        ordered_cols = [c for c in ordered_cols if c in out_df.columns]
        out_df = out_df[ordered_cols]
        out_path = os.path.join(out_dir, 'comparison_core_metrics.csv')
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {os.path.basename(out_path)}")

    return out_df


def save_outcome_breakdown_table(df, out_dir):
    """Save a compact outcome-breakdown table for failure-mode analysis."""
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for algo, grp in df.groupby('algo'):
        total = len(grp)
        n_success = int((grp['success'] == 1).sum())
        n_path = int((grp['path_found'] == 1).sum()) if 'path_found' in grp.columns else 0
        n_path_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0)).sum()) if 'path_found' in grp.columns else 0
        n_path_tw_fail = int(((grp['path_found'] == 1) & (grp['success'] == 0) & (grp['tw_viol_positive'] == 1)).sum()) if 'tw_viol_positive' in grp.columns else 0
        n_no_path = int((grp['path_found'] == 0).sum()) if 'path_found' in grp.columns else total - n_path

        path_fail_subset = grp[(grp['path_found'] == 1) & (grp['success'] == 0)] if 'path_found' in grp.columns else pd.DataFrame(columns=grp.columns)
        tw_fail_subset = grp[(grp['path_found'] == 1) & (grp['success'] == 0) & (grp['tw_viol_positive'] == 1)] if 'tw_viol_positive' in grp.columns else pd.DataFrame(columns=grp.columns)

        rows.append({
            'algo': algo,
            'total_trials': int(total),
            'full_success_rate_pct': (n_success / total * 100.0) if total > 0 else np.nan,
            'path_found_rate_pct': (n_path / total * 100.0) if total > 0 else np.nan,
            'path_found_but_final_fail_rate_pct': (n_path_fail / total * 100.0) if total > 0 else np.nan,
            'path_found_but_tw_fail_rate_pct': (n_path_tw_fail / total * 100.0) if total > 0 else np.nan,
            'no_path_rate_pct': (n_no_path / total * 100.0) if total > 0 else np.nan,
            'temporal_success_given_path_pct': (n_success / n_path * 100.0) if n_path > 0 else np.nan,
            'tw_viol_rate_given_path_pct': tw_violation_split(grp.loc[grp['path_found'] == 1, 'time_window_violation'])[0] if ('path_found' in grp.columns and 'time_window_violation' in grp.columns and n_path > 0) else np.nan,
            'tw_viol_mean_mag_path_fail_s': float(pd.to_numeric(path_fail_subset['time_window_violation'], errors='coerce').dropna().mean()) if not path_fail_subset.empty else np.nan,
            'tw_viol_mean_mag_tw_fail_s': float(pd.to_numeric(tw_fail_subset['time_window_violation'], errors='coerce').dropna().mean()) if not tw_fail_subset.empty else np.nan,
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_path = os.path.join(out_dir, 'comparison_outcome_breakdown.csv')
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {os.path.basename(out_path)}")
    return out_df


def plot_metric_boxplots(df, out_dir, metrics_to_plot, benchmark_info=None):
    """
    Plot distributions of Raw Trial Data.
    benchmark_info: dict with {'t_ref', 'tmin', 'tmax'} or None
    """
    if df.empty: return

    # Academic Metric Labels
    metric_labels = {
        'runtime_s': 'Computation Time (s)',
        'path_len': 'Path Length (m)',
        'node_count': 'Node Count',
        'arrival_t': 'Arrival Time (s)',
        'spatial_err': 'Spatial Error (m)',
        'time_err': 'Time Error (s)',
        'total_err': 'Total Error',
        'time_window_violation': 'Time-Window Violation (s)',
        'first_sol_time': 'First Solution Time (s)',
        'smoothness': 'Path Smoothness'
    }

    # Algorithm names for labels
    algos = sorted(df['algo'].unique())
    algo_labels = [_algo_display_name(a) for a in algos]

    plot_targets = []
    if 'runtime_s' in metrics_to_plot:
        plot_targets.append(('runtime_s', 'Runtime (All)', 'all'))
        plot_targets.append(('runtime_s', 'Runtime (Success Only)', 'success'))
        
    for m in metrics_to_plot:
        if m == 'runtime_s': continue
        if m in ['spatial_err', 'time_err', 'total_err', 'time_window_violation']:
            plot_targets.append((m, m, 'all'))
        else:
            plot_targets.append((m, m, 'success'))
            
    for (col_name, title_suffix, criteria) in plot_targets:
        if col_name not in df.columns: continue
        
        plt.figure(figsize=(6.2, 3.5))
        
        data_to_plot = []
        positions_to_plot = []

        for i, algo in enumerate(algos, start=1):
            subset_df = df[df['algo'] == algo]

            if criteria == 'success':
                vals = subset_df[subset_df['success'] == 1][col_name].dropna()
            else:
                vals = subset_df[col_name].dropna()

            if not vals.empty:
                data_to_plot.append(vals.values)
                positions_to_plot.append(i)

        if not data_to_plot:
            plt.close()
            continue

        plt.boxplot(
            data_to_plot,
            positions=positions_to_plot,
            widths=0.55,
            patch_artist=True
        )

        # Keep all algorithms on the x-axis, even if an algorithm has no successful trials.
        plt.xticks(
            range(1, len(algos) + 1),
            algo_labels,
            fontsize=13
        )

        plt.xlim(0.5, len(algos) + 0.5)
        
        plt.title(f"{col_name.replace('_', ' ').title()} - {criteria.title()}", fontsize=16)
        plt.ylabel(metric_labels.get(col_name, col_name.replace('_', ' ').title()), fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save Plain Version (No Reference Lines)
        plt.tight_layout()
        fname_plain = f"boxplot_{col_name}_{criteria}_plain.png"
        plt.savefig(os.path.join(out_dir, fname_plain), dpi=220)

        # Add Reference Lines for Arrival Time
        if col_name == 'arrival_t':
            t_ref = benchmark_info.get('t_ref') if benchmark_info else None
            tmin = benchmark_info.get('tmin') if benchmark_info else None
            tmax = benchmark_info.get('tmax') if benchmark_info else None
            
            # Fallback to data extraction
            if t_ref is None and 't_ref' in df.columns:
                t_ref_vals = df['t_ref'].dropna().unique()
                if len(t_ref_vals) >= 1: t_ref = t_ref_vals[0]
            
            if (tmin is None or tmax is None):
                if 'tmin' in df.columns and 'tmax' in df.columns:
                    t_min_vals = df['tmin'].dropna().unique()
                    t_max_vals = df['tmax'].dropna().unique()
                    if len(t_min_vals) >= 1: tmin = tmin or t_min_vals[0]
                    if len(t_max_vals) >= 1: tmax = tmax or t_max_vals[0]
                
            lines_added = False
            if t_ref is not None:
                plt.axhline(y=t_ref, color='r', linestyle='--', alpha=0.6, label='t_ref')
                lines_added = True
            if tmin is not None and tmax is not None:
                plt.axhline(y=tmin, color='g', linestyle=':', alpha=0.5, label='t_window')
                plt.axhline(y=tmax, color='g', linestyle=':', alpha=0.5)
                lines_added = True
            
            if lines_added:
                # Adjust Y-limits to ensure reference lines are visible and data isn't cropped
                all_vals = []
                for d in data_to_plot:
                    if len(d) > 0: all_vals.extend(d)
                
                if all_vals:
                    data_min, data_max = min(all_vals), max(all_vals)
                    ref_pts = [v for v in [t_ref, tmin, tmax] if v is not None]
                    
                    y_min = min([data_min] + ref_pts)
                    y_max = max([data_max] + ref_pts)
                    
                    # Add 15% padding
                    pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.5
                    plt.ylim(y_min - pad, y_max + pad)

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

                # Save Reference Version
                plt.tight_layout()
                fname_ref = f"boxplot_{col_name}_{criteria}.png"
                plt.savefig(os.path.join(out_dir, fname_ref), dpi=120)
                print(f"  Saved plot: {fname_ref} (with window)")

        plt.close()
        print(f"  Saved plot: {fname_plain}")


# ===========================================================================
# Additional paper-appeal plot functions
# ===========================================================================

def _algo_display_name(algo):
    """
    Rename 'TWA-RRT*' to 'TWA-RRT' for display and 
    ensure algorithm display name ends with '*' (e.g. TB-RRT*).
    """
    # 1. Rename WISER to TWA (case-insensitive)
    name = re.sub(r'WISER-RRT', 'TWA-RRT', algo, flags=re.IGNORECASE)
    
    # 2. Add asterisk if missing
    if name.endswith("RRT") and not name.endswith("RRT*") and "*" not in name:
        return name + "*"
    return name


def plot_ecdf_runtime(df, out_dir, only_success=False):
    """
    Plot #1: Runtime ECDF — one line per algorithm.
    """
    if 'runtime_s' not in df.columns or df['runtime_s'].dropna().empty:
        print("  [Warning] runtime_s column missing/empty → skipping ECDF runtime plot.")
        return

    subset = df.copy()
    suffix = "all"
    if only_success:
        subset = subset[subset['success'] == 1]
        suffix = "success"

    # Global x_max for extending lines beyond last data point
    all_rt = subset['runtime_s'].dropna()
    x_max = all_rt.max() * 1.05 if not all_rt.empty else 1.0

    plt.figure(figsize=(8, 5))
    for algo in sorted(subset['algo'].unique()):
        vals = subset.loc[subset['algo'] == algo, 'runtime_s'].dropna().sort_values().values
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        # Append (x_max, 1.0) so the line extends at y=1 to the right edge
        vals_ext = np.append(vals, x_max)
        y_ext = np.append(y, 1.0)
        label = f"{_algo_display_name(algo)}"
        plt.step(vals_ext, y_ext, where='post', label=label)

    plt.xlabel('Computation Time (s)')
    plt.ylabel('ECDF')
    # plt.title('Runtime ECDF')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fname = f"ecdf_runtime_{suffix}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def plot_deadline_curve(df, out_dir, deadlines):
    """
    Plot #2: Deadline satisfaction curve — P(runtime <= t) vs t.
    """
    if 'runtime_s' not in df.columns or df['runtime_s'].dropna().empty:
        print("  [Warning] runtime_s column missing/empty → skipping deadline curve.")
        return

    plt.figure(figsize=(8, 5))
    for algo in sorted(df['algo'].unique()):
        rts = df.loc[df['algo'] == algo, 'runtime_s'].dropna().values
        n = len(rts)
        if n == 0:
            continue
        probs = [np.sum(rts <= t) / n for t in deadlines]
        label = f"{_algo_display_name(algo)}"
        plt.plot(deadlines, probs, marker='o', markersize=4, label=label)

    plt.xlabel('Deadline (s)')
    plt.ylabel('P(runtime ≤ deadline)')
    plt.ylim(-0.05, 1.05)
    plt.title('Deadline Satisfaction Curve')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fname = "deadline_runtime_curve.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def compute_budget_table(df, out_dir, budgets, eps=1e-12):
    """Compute fair compute-budget comparison table.

    Interpretation:
      - Any trial with runtime_s > budget is treated as "not available under budget".
      - success_under_budget(%) = P(success==1 AND runtime<=budget) over all trials.
      - runtime_coverage(%) = P(runtime<=budget) over all trials.
      - Error metrics are computed on (success==1 AND runtime<=budget) trials.
      - Time-window violation is split into rate + magnitude, computed on the same subset.

    Outputs:
      - compute_budget_table.csv
    """
    if df.empty or 'runtime_s' not in df.columns:
        print("  [Warning] No runtime_s data → skipping compute-budget table.")
        return pd.DataFrame()

    rows = []
    for algo, grp in df.groupby('algo'):
        total = len(grp)
        if total == 0:
            continue

        grp_rt = grp[grp['runtime_s'].notna()]

        for b in budgets:
            within = grp_rt[grp_rt['runtime_s'] <= b]
            n_within = len(within)

            succ_within = within[within['success'] == 1]
            n_succ_within = len(succ_within)
            path_within = within[within['path_found'] == 1] if 'path_found' in within.columns else pd.DataFrame(columns=within.columns)
            n_path_within = len(path_within)
            n_path_fail_within = int(((within['path_found'] == 1) & (within['success'] == 0)).sum()) if 'path_found' in within.columns else 0

            runtime_coverage = (n_within / total * 100.0) if total > 0 else np.nan
            success_under_budget = (n_succ_within / total * 100.0) if total > 0 else np.nan

            # Errors on success-within subset
            spatial_mean = succ_within['spatial_err'].mean() if 'spatial_err' in succ_within else np.nan
            time_mean = succ_within['time_err'].mean() if 'time_err' in succ_within else np.nan
            totalerr_mean = succ_within['total_err'].mean() if 'total_err' in succ_within else np.nan

            # TW violation split
            if 'time_window_violation' in within.columns:
                tw_rate_all, tw_mean_all, tw_max_all = tw_violation_split(within['time_window_violation'], eps=eps)
                tw_rate_s, tw_mean_s, tw_max_s = tw_violation_split(succ_within['time_window_violation'], eps=eps) if n_succ_within > 0 else (np.nan, np.nan, np.nan)
            else:
                tw_rate_all = tw_mean_all = tw_max_all = np.nan
                tw_rate_s = tw_mean_s = tw_max_s = np.nan

            rows.append({
                'algo': algo,
                'budget_s': float(b),
                'total_trials': int(total),
                'n_within_budget': int(n_within),
                'runtime_coverage_pct': float(runtime_coverage),
                'n_success_within_budget': int(n_succ_within),
                'success_under_budget_pct': float(success_under_budget),
                'n_path_within_budget': int(n_path_within),
                'path_found_within_budget_pct': float(n_path_within / total * 100.0) if total > 0 else np.nan,
                'path_found_but_fail_within_budget_pct': float(n_path_fail_within / total * 100.0) if total > 0 else np.nan,
                'temporal_success_given_path_within_pct': float(n_succ_within / n_path_within * 100.0) if n_path_within > 0 else np.nan,

                'spatial_err_mean_succ_within': float(spatial_mean) if pd.notna(spatial_mean) else np.nan,
                'time_err_mean_succ_within': float(time_mean) if pd.notna(time_mean) else np.nan,
                'total_err_mean_succ_within': float(totalerr_mean) if pd.notna(totalerr_mean) else np.nan,

                'tw_viol_rate_all_within_pct': float(tw_rate_all) if pd.notna(tw_rate_all) else np.nan,
                'tw_viol_mean_mag_all_within': float(tw_mean_all) if pd.notna(tw_mean_all) else np.nan,
                'tw_viol_max_mag_all_within': float(tw_max_all) if pd.notna(tw_max_all) else np.nan,

                'tw_viol_rate_succ_within_pct': float(tw_rate_s) if pd.notna(tw_rate_s) else np.nan,
                'tw_viol_mean_mag_succ_within': float(tw_mean_s) if pd.notna(tw_mean_s) else np.nan,
                'tw_viol_max_mag_succ_within': float(tw_max_s) if pd.notna(tw_max_s) else np.nan,
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        out_path = os.path.join(out_dir, "compute_budget_table.csv")
        table.to_csv(out_path, index=False)
        print(f"  Saved table: {os.path.basename(out_path)}")

    return table


def plot_budget_success_curve(budget_table, out_dir):
    """Plot: Success-under-budget curve."""
    if budget_table is None or budget_table.empty:
        return

    plt.figure(figsize=(8, 5))
    for algo in sorted(budget_table['algo'].unique()):
        a = budget_table[budget_table['algo'] == algo].sort_values('budget_s')
        plt.plot(a['budget_s'], a['success_under_budget_pct'], marker='o', markersize=4,
                 label=_algo_display_name(algo))

    plt.xlabel('Compute Budget (s)')
    plt.ylabel('P(success ∧ runtime ≤ budget) [%]')
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fname = "budget_success_curve.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def plot_budget_tw_violation_curve(budget_table, out_dir, use_success=True):
    """Plot: Time-window violation rate under compute budget (within-budget trials)."""
    if budget_table is None or budget_table.empty:
        return

    col = 'tw_viol_rate_succ_within_pct' if use_success else 'tw_viol_rate_all_within_pct'
    ylabel = 'TW violation rate in success-within [%]' if use_success else 'TW violation rate in all within-budget trials [%]'

    plt.figure(figsize=(8, 5))
    for algo in sorted(budget_table['algo'].unique()):
        a = budget_table[budget_table['algo'] == algo].sort_values('budget_s')
        plt.plot(a['budget_s'], a[col], marker='o', markersize=4,
                 label=_algo_display_name(algo))

    plt.xlabel('Compute Budget (s)')
    plt.ylabel(ylabel)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fname = "budget_tw_violation_rate_curve.png" if use_success else "budget_tw_violation_rate_curve_all.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def plot_budget_error_curves(budget_table, out_dir):
    """Plot: Mean spatial/time error under compute budget (success-within)."""
    if budget_table is None or budget_table.empty:
        return

    # Spatial error curve
    plt.figure(figsize=(8, 5))
    for algo in sorted(budget_table['algo'].unique()):
        a = budget_table[budget_table['algo'] == algo].sort_values('budget_s')
        plt.plot(a['budget_s'], a['spatial_err_mean_succ_within'], marker='o', markersize=4,
                 label=_algo_display_name(algo))
    plt.xlabel('Compute Budget (s)')
    plt.ylabel('Mean Spatial Error (success ∧ within-budget)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fname = "budget_spatial_error_curve.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")

    # Time error curve
    plt.figure(figsize=(8, 5))
    for algo in sorted(budget_table['algo'].unique()):
        a = budget_table[budget_table['algo'] == algo].sort_values('budget_s')
        plt.plot(a['budget_s'], a['time_err_mean_succ_within'], marker='o', markersize=4,
                 label=_algo_display_name(algo))
    plt.xlabel('Compute Budget (s)')
    plt.ylabel('Mean Time Error (success ∧ within-budget)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fname = "budget_time_error_curve.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def plot_ecdf_first_solution(df, out_dir):
    """
    Plot #3: First-solution time ECDF (success trials only).
    """
    if 'first_sol_time' not in df.columns or df['first_sol_time'].dropna().empty:
        print("  [Warning] first_sol_time column missing/all-NaN → skipping first-solution ECDF.")
        return

    subset = df[(df['success'] == 1) & df['first_sol_time'].notna()]
    if subset.empty:
        print("  [Warning] No success trials with first_sol_time → skipping first-solution ECDF.")
        return

    # Global x_max for extending lines beyond last data point
    all_fst = subset['first_sol_time'].dropna()
    x_max = all_fst.max() * 1.05 if not all_fst.empty else 1.0

    plt.figure(figsize=(8, 5))
    for algo in sorted(subset['algo'].unique()):
        vals = subset.loc[subset['algo'] == algo, 'first_sol_time'].dropna().sort_values().values
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        # Append (x_max, 1.0) so the line extends at y=1 to the right edge
        vals_ext = np.append(vals, x_max)
        y_ext = np.append(y, 1.0)
        label = f"{_algo_display_name(algo)}"
        plt.step(vals_ext, y_ext, where='post', label=label)

    plt.xlabel('First Solution Time (s)')
    plt.ylabel('ECDF')
    # plt.title('First-Solution Time ECDF (Success Only)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fname = "ecdf_first_solution_success.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def _compute_pareto_front(x, y):
    """
    2-D nondominated front (minimise both x and y).
    Returns sorted arrays (x_front, y_front).
    """
    pts = np.column_stack((x, y))
    # Sort by x, then by y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    front = [pts[0]]
    for p in pts[1:]:
        if p[1] <= front[-1][1]:  # non-dominated
            front.append(p)
    front = np.array(front)
    return front[:, 0], front[:, 1]


def plot_pareto_scatter(df, out_dir):
    """
    Plot #4: Pareto scatter — runtime_s vs path_len (success only).
    """
    if 'runtime_s' not in df.columns or 'path_len' not in df.columns:
        print("  [Warning] runtime_s or path_len missing → skipping Pareto scatter.")
        return

    subset = df[(df['success'] == 1) &
                df['runtime_s'].notna() &
                df['path_len'].notna()]
    if subset.empty:
        print("  [Warning] No success trials with runtime_s & path_len → skipping Pareto scatter.")
        return

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    plt.figure(figsize=(8, 6))

    for i, algo in enumerate(sorted(subset['algo'].unique())):
        a_df = subset[subset['algo'] == algo]
        rx = a_df['runtime_s'].values
        ry = a_df['path_len'].values
        mk = markers[i % len(markers)]
        lbl = f"{_algo_display_name(algo)}"
        plt.scatter(rx, ry, marker=mk, s=25, alpha=0.6, label=lbl)

        # Pareto front line
        if len(rx) >= 2:
            try:
                fx, fy = _compute_pareto_front(rx, ry)
                if len(fx) >= 2:
                    plt.plot(fx, fy, linewidth=1.0, alpha=0.7)
            except Exception:
                pass  # skip front on error

    plt.xlabel('Computation Time (s)')
    plt.ylabel('Path Length (m)')
    plt.title('Pareto: Computation Time vs Path Length (Success Only)')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fname = "pareto_runtime_pathlen_success.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def plot_success_only_boxplots(df, out_dir):
    """
    Bonus: success-only quality boxplots for error / quality metrics.
    """
    targets = [
        ('spatial_err',           'Spatial Error (m)'),
        ('time_err',              'Time Error (s)'),
        ('time_window_violation', 'Time-Window Violation'),
        ('smoothness',            'Path Smoothness'),
    ]
    succ_df = df[df['success'] == 1]
    if succ_df.empty:
        print("  [Warning] No success trials → skipping success-only boxplots.")
        return

    for col, ylabel in targets:
        if col not in succ_df.columns or succ_df[col].dropna().empty:
            print(f"  [Warning] {col} missing/empty in success trials → skipping boxplot.")
            continue

        plt.figure(figsize=(6.2, 3.5))
        algos = sorted(df['algo'].unique())
        data = []
        positions = []

        for i, algo in enumerate(algos, start=1):
            vals = succ_df.loc[succ_df['algo'] == algo, col].dropna()

            if vals.empty:
                continue

            data.append(vals.values)
            positions.append(i)

        if not data:
            plt.close()
            continue

        plt.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True
        )

        plt.xticks(
            range(1, len(algos) + 1),
            [_algo_display_name(a) for a in algos],
            fontsize=13
        )

        plt.xlim(0.5, len(algos) + 0.5)

        plt.ylabel(ylabel, fontsize=15)
        # plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fname = f"boxplot_{col}_success.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()
        print(f"  Saved plot: {fname}")

def plot_success_rate(df, out_dir):
    """Bar plot of overall success rate for each algorithm."""
    if df is None or df.empty or 'success' not in df.columns:
        return

    algos = sorted(df['algo'].unique())

    labels = []
    rates = []

    for algo in algos:
        grp = df[df['algo'] == algo]

        if len(grp) == 0:
            continue

        rate = float((grp['success'] == 1).mean() * 100.0)

        labels.append(_algo_display_name(algo))
        rates.append(rate)

    if not rates:
        return

    x = np.arange(len(labels))

    plt.figure(figsize=(6.2, 3.5))
    plt.bar(x, rates)

    plt.xticks(x, labels, fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel('Success Rate (%)', fontsize=15)
    plt.ylim(0, 100)

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    fname = 'success_rate.png'
    plt.savefig(
        os.path.join(out_dir, fname),
        dpi=220
    )

    plt.close()

    print(f"  Saved plot: {fname}")


def plot_outcome_breakdown_stacked(df, out_dir):
    """Stacked bar plot: full success vs path-found-but-fail vs no-path."""
    if df is None or df.empty or 'path_found' not in df.columns:
        return

    rows = []
    for algo, grp in sorted(df.groupby('algo'), key=lambda x: x[0]):
        total = len(grp)
        if total == 0:
            continue
        full_success = float((grp['success'] == 1).sum() / total * 100.0)
        path_fail = float(((grp['path_found'] == 1) & (grp['success'] == 0)).sum() / total * 100.0)
        no_path = float((grp['path_found'] == 0).sum() / total * 100.0)
        rows.append((algo, full_success, path_fail, no_path))

    if not rows:
        return

    labels = [_algo_display_name(r[0]) for r in rows]
    full_success = np.array([r[1] for r in rows])
    path_fail = np.array([r[2] for r in rows])
    no_path = np.array([r[3] for r in rows])
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5.5))
    plt.bar(x, full_success, label='Full success')
    plt.bar(x, path_fail, bottom=full_success, label='Path found but final fail')
    plt.bar(x, no_path, bottom=full_success + path_fail, label='No path')
    plt.xticks(x, labels)
    plt.ylabel('Trial Ratio (%)')
    plt.ylim(0, 100)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fname = 'stacked_outcome_breakdown.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


def main():
    args = parse_arguments()
    
    # Parse multiple cases (comma or space separated)
    case_list = [c.strip() for c in args.case.replace(',', ' ').split() if c.strip()]
    
    if not case_list:
        print("Error: No valid cases provided.")
        sys.exit(1)
    
    # Keep this batch separate from previous r5/r6 outputs.
    args.out_root = os.path.join(args.out_root, datetime.datetime.now().strftime('batch_%Y%m%d_%H%M%S_%f'))
    os.makedirs(args.out_root, exist_ok=False)
    case_outputs = []
    # Process each case
    for case_id in case_list:
        print(f"\n{'='*60}")
        print(f"Processing Case: {case_id}")
        print(f"{'='*60}\n")
        
        case_outputs.append(process_single_case(args, case_id))

    for filename in ('comparison_core_metrics.csv', 'comparison_stats_pooled.csv',
                     'comparison_trials_pooled.csv', 'comparison_outcome_breakdown.csv'):
        tables = []
        for case_id, folder in zip(case_list, case_outputs):
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                table = pd.read_csv(path)
                table['case_id'] = case_id
                tables.append(table)
        if tables:
            merged = pd.concat(tables, ignore_index=True)
            merged.to_csv(os.path.join(args.out_root, filename.replace('.csv', '_all_cases.csv')), index=False)
    print(f'Batch results: {args.out_root}')
    
    print(f"\nAll cases processed.")


def process_single_case(args, case_id):
    """Process a single case."""
    # 1. Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_name = f"{timestamp}_{case_id}"
    final_out_dir = os.path.join(args.out_root, out_dir_name)
    os.makedirs(final_out_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" RRT Results Comparison")
    print(f" Case: {case_id}")
    print(f" Run Mode: {args.run_mode} (Ignored if *_run override provided)")
    print(f" Output: {final_out_dir}")
    print(f"==================================================\n")
    
    # 1.5 Load Benchmark Case Info (Ground Truth for plots)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_info = load_benchmark_case_info(case_id, script_dir)
    if benchmark_info:
        print(f"Loaded benchmark info for {case_id}: t_ref={benchmark_info['t_ref']}, t_window=[{benchmark_info['tmin']}, {benchmark_info['tmax']}]")
    else:
        print(f"Warning: No benchmark JSON found for case {case_id}. Will try to extract from data.")
    
    # 2. Resolve Paths (with case_id substitution for directory discovery)
    algo_paths = resolve_algo_paths(args, case_id)
    
    all_dfs = []
    selected_sources = []
    
    for algo_key, info in algo_paths.items():
        print(f"Processing {algo_key}...")
        
        # A) Explicit Run Override?
        if info['run_override']:
            print(f"  -> Using override path: {info['run_override']}")
            run_list = discover_one_summary_from_override(info['run_override'], algo_key, case_id)
        # B) Standard Discovery
        else:
            print(f"  -> searching in: {info['dir']}")
            run_list = discover_run_summaries(
                base_dir=info['dir'], 
                algo_key=algo_key, 
                case_id=case_id, 
                run_mode=args.run_mode, 
                topN=args.topN
            )
            
        if not run_list:
            continue
            
        print(f"  -> Found {len(run_list)} run(s).")
        
        # Load Data
        for rinfo in run_list:
            try:
                df = pd.read_csv(rinfo['summary_csv'])
                df = enrich_hpo_summary(df, rinfo)
                norm_df = normalize_metrics(df, rinfo)
                all_dfs.append(norm_df)
                selected_sources.append(dict(rinfo, trials=len(norm_df),
                    seeds=sorted(pd.to_numeric(norm_df['seed'], errors='coerce').dropna().astype(int).unique().tolist())))
            except Exception as e:
                raise RuntimeError(f"Failed to read {rinfo['summary_csv']}: {e}") from e

    if not all_dfs:
        print("\nNo valid data found for any algorithm. Exiting.")
        sys.exit(0)
        
    combined_df = add_outcome_flags(pd.concat(all_dfs, ignore_index=True))
    with open(os.path.join(final_out_dir, 'selected_runs.json'), 'w', encoding='utf-8') as f:
        json.dump({'case_id': case_id, 'run_mode': args.run_mode, 'runs': selected_sources,
                   'notes': ['HPO path_found uses raw reached_goal; legacy missing flags use r5 evidence fallback.',
                             'HPO first_sol_time is geometric completion time; runtime measurements were collected at different times.']},
                  f, indent=2, ensure_ascii=False)
    
    # Save combined raw
    if args.save_raw:
        combined_df.to_csv(os.path.join(final_out_dir, "combined_raw.csv"), index=False)
        print(f"\nSaved combined raw data.")

    # --- Save trial-level pooled CSV (always) ---
    trials_rename = {
        'trial': 'trial_id',
        'first_sol_time': 'first_sol_s',
        'path_len': 'path_len_m',
        'spatial_err': 'spatial_err_m',
        'time_err': 'time_err_s',
        'time_window_violation': 'tw_violation',
    }
    trials_df = combined_df.rename(columns=trials_rename)
    trials_cols = [
        'algo', 'trial_id', 'seed', 'run_id', 'case_id', 'success', 'path_found', 'tw_satisfied',
        'path_found_but_final_fail', 'path_found_but_tw_fail', 'no_path', 'outcome_category',
        'runtime_s', 'first_sol_s', 'path_len_m', 'spatial_err_m', 'time_err_s',
        'tw_violation', 'node_count', 'arrival_t', 'smoothness'
    ]
    # Keep only columns that exist
    trials_cols = [c for c in trials_cols if c in trials_df.columns]
    trials_df[trials_cols].to_csv(
        os.path.join(final_out_dir, "comparison_trials_pooled.csv"), index=False)
    print(f"  Saved comparison_trials_pooled.csv ({len(trials_df)} rows)")

    # 3. Calculate Stats
    print("\nCalculating Statistics...")
    # Note: calculate_stats returns the run-level stats DF used for CSVs
    # but for plotting we will re-derive it inside plot_metric_boxplots 
    # (or we could return it here and pass it, but following original design is fine)
    _ = calculate_stats(combined_df, final_out_dir)
    _ = save_core_metrics_table(combined_df, final_out_dir)
    _ = save_outcome_breakdown_table(combined_df, final_out_dir)
    print(f"  Saved statistics CSVs to {final_out_dir}")
    
    # 4. Plots (existing boxplots)
    if not args.no_plots:
        print("\nGenerating Boxplots...")
        # Metrics to include in standard summary boxplots
        default_plot_metrics = [
            'runtime_s', 'path_len', 'node_count', 'arrival_t', 
            'spatial_err', 'time_err', 'total_err', 'time_window_violation',
            'first_sol_time', 'smoothness'
        ]
        
        # Combined request (either default or explicitly provided)
        metrics_to_plot = default_plot_metrics
        if args.metrics:
            metrics_to_plot = [m.strip() for m in args.metrics.split(',') if m.strip()]
            
        plot_metric_boxplots(combined_df, final_out_dir, metrics_to_plot, benchmark_info=benchmark_info)

    # 5. Additional paper-appeal plots
    if not args.no_plots and not args.no_extra_plots:
        print("\nGenerating Additional Paper-Appeal Plots...")
        # Parse deadlines
        try:
            deadlines = sorted([float(d.strip()) for d in args.deadlines.split(',') if d.strip()])
        except ValueError:
            print("  [Warning] Could not parse --deadlines, using defaults.")
            deadlines = [0.5, 1.0, 2.0, 5.0, 10.0]

        plot_ecdf_runtime(combined_df, final_out_dir, only_success=args.ecdf_only_success)
        plot_deadline_curve(combined_df, final_out_dir, deadlines)

        # Compute-budget fair comparison (success / violation / error under fixed time budgets)
        budget_table = compute_budget_table(combined_df, final_out_dir, deadlines)
        plot_budget_success_curve(budget_table, final_out_dir)
        plot_budget_tw_violation_curve(budget_table, final_out_dir, use_success=False)
        plot_budget_tw_violation_curve(budget_table, final_out_dir, use_success=True)
        plot_budget_error_curves(budget_table, final_out_dir)

        plot_ecdf_first_solution(combined_df, final_out_dir)
        plot_pareto_scatter(combined_df, final_out_dir)

        plot_success_rate(combined_df, final_out_dir)
        
        plot_success_only_boxplots(combined_df, final_out_dir)
        plot_outcome_breakdown_stacked(combined_df, final_out_dir)

    print(f"\nDone.")
    return final_out_dir

if __name__ == "__main__":
    main()
