#!/usr/bin/env python3
"""Run C1/C2 on frozen 2018–2020 music blocks without touching old results."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from saet_repro.extension_transfer import Runner, BASE_CONFIGS, ORIGINS, FINAL_SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data_extensions/v1/music')
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'results_extensions/v1/stage_c_music')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/extensions_v1.json')
    parser.add_argument('--wait-data', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    for mode in ('freeze-only', 'historical-only', 'hpo-only', 'final-only'):
        modes.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / 'runner.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another C runner owns this output directory')
        lock.write(str(os.getpid()))
        lock.flush()
        runner = Runner(args.data_dir, args.result_dir, args.protocol, args.wait_data)
        runner.freeze()
        if args.freeze_only:
            return
        if not args.final_only:
            ridges = {(base, year): runner.fit_ridges(base, year)
                      for base in BASE_CONFIGS for year in ORIGINS}
            if args.historical_only:
                return
            for base in BASE_CONFIGS:
                runner.search(base, ridges[base, 2018])
            if args.hpo_only:
                return
        # Gate is written only after every historical/HPO dependency is verified,
        # and immediately before the first call that opens a d0 archive.
        ridges, selected = runner.open_final_queries()
        for year in ORIGINS:
            data = runner.load_prefix(year, 0)
            runner.activity_evaluation(year, data)
            for base in BASE_CONFIGS:
                for seed in FINAL_SEEDS:
                    runner.final_evaluation(base, year, seed, ridges[base, year], selected[base], data)
            del data


if __name__ == '__main__':
    main()
