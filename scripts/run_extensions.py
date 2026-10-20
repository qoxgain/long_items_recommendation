#!/usr/bin/env python3
"""Run the frozen supplemental stages through their independently checked CLIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('a', 'b', 'c', 'all'), default='all')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/extensions_v1.json')
    parser.add_argument('--list-tasks', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--hpo-only', action='store_true')
    modes.add_argument('--final-only', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.pilot and args.stage != 'b':
        parser.error('--pilot is defined for --stage b only')
    if args.stage in ('a', 'all') and (args.hpo_only or args.final_only):
        parser.error('select --stage b or c for separate historical/final phases')
    protocol = json.loads(args.protocol.read_text())
    stages = ('a', 'b', 'c') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if args.list_tasks and stage != 'b':
            key = 'stage_a' if stage == 'a' else 'stage_c_music'
            print(json.dumps(protocol[key], indent=2, ensure_ascii=False))
            continue
        scripts = ['analyze_extensions.py'] if stage == 'a' else (
            ['run_extension_cf.py', 'run_extension_targets.py'] if stage == 'b' else ['run_extension_transfer.py'])
        for script in scripts:
            command = [sys.executable, str(ROOT / 'scripts' / script), '--protocol', str(args.protocol.resolve())]
            if stage == 'b':
                for option in ('list_tasks', 'pilot', 'hpo_only', 'final_only', 'retry_failed'):
                    if getattr(args, option):
                        command.append('--' + option.replace('_', '-'))
            elif stage == 'c':
                for option in ('hpo_only', 'final_only'):
                    if getattr(args, option):
                        command.append('--' + option.replace('_', '-'))
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
