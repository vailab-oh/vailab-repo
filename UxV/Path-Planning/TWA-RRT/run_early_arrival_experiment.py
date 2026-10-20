"""Sequential M5 benchmark with frozen inputs, provenance and isolated outputs."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    'tb_rrt': 'baselines/tb_rrt.py',
    'st_rrt': 'baselines/st_rrt.py',
    'stl_rrt': 'baselines/stl_rrt.py',
    'wiser_rrt': 'twa_rrt.py',
    'hpo_rrt_r1': 'baselines/hpo_rrt/run_hpo_benchmark.py',
}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    batch = ROOT / 'output' / 'early_arrival_experiments' / dt.datetime.now().strftime('M5_100_%Y%m%d_%H%M%S')
    batch.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'benchmarks/benchmark_cases_M1_with_early_arrival.json'
    original = json.loads(source.read_text(encoding='utf-8'))
    case = next(c for c in original['cases'] if c['case_id'] == 'early_arrival')
    case['random']['num_trials'] = 100
    config = batch / 'benchmark_M5_100.json'
    write(config, {'meta': {'source': str(source), 'change': 'M5 only; num_trials 5 -> 100 per user'}, 'cases': [case]})
    paths = list(SCRIPTS.values()) + ['baselines/hpo_rrt/hpo_rrt_star.py', 'baselines/hpo_rrt/benchmark_adapter.py', source.relative_to(ROOT).as_posix()]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    manifest = {'status': 'running', 'started_at': dt.datetime.now().astimezone().isoformat(),
                'hostname': socket.gethostname(), 'platform': platform.platform(),
                'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
                'python_executable': sys.executable, 'python_version': sys.version,
                'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib']},
                'source_sha256': hashes, 'config': str(config), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'case_id': case['case_id'], 'trials_per_model': 100, 'seed_base': case['random']['seed_base'],
                'protocol': 'Sequential model processes; original algorithms and native timers/termination; plots enabled; no trial retries.',
                'runs': []}
    write(batch/'manifest.json', manifest)
    print(f'BATCH={batch}', flush=True)
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(batch/'mpl_cache'), PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    for algo, script in SCRIPTS.items():
        cmd = [sys.executable, str(ROOT/script), '--cases', str(config), '--case_id', case['case_id'], '--out_dir', str(batch/'results')]
        entry = {'algorithm': algo, 'command_argv': cmd, 'started_at': dt.datetime.now().astimezone().isoformat(), 'status':'running'}
        manifest['runs'].append(entry)
        write(batch/'manifest.json', manifest)
        print(f'START {algo}', flush=True)
        t0 = time.perf_counter()
        with (batch/f'{algo}.log').open('w', encoding='utf-8') as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=proc.returncode, process_wall_s=time.perf_counter()-t0, finished_at=dt.datetime.now().astimezone().isoformat())
        files = list((batch/'results'/case['case_id']/algo).glob('run_*/summary.csv'))
        rows = []
        if len(files) == 1:
            with files[0].open(newline='',encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
        expected = list(range(case['random']['seed_base'],case['random']['seed_base']+100))
        good = proc.returncode == 0 and len(rows)==100 and sorted(int(r['seed']) for r in rows)==expected
        entry.update(status='complete' if good else 'incomplete', trials=len(rows), summary_csv=str(files[0]) if files else None)
        write(batch/'manifest.json', manifest)
        print(f"DONE {algo}: {entry['status']}, {len(rows)} trials, {entry['process_wall_s']:.1f}s", flush=True)
        if not good:
            manifest['status']='incomplete'
            write(batch/'manifest.json', manifest)
            raise SystemExit(1)
    manifest.update(status='complete',finished_at=dt.datetime.now().astimezone().isoformat(),
                    sources_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items()))
    write(batch/'manifest.json', manifest)
    print('COMPLETE '+str(batch), flush=True)


if __name__ == '__main__':
    main()
