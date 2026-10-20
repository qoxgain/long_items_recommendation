#!/usr/bin/env python3
"""Build the prespecified new-time prefixes in an isolated directory."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saet_repro.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/extensions_v1.json')
    parser.add_argument('--list-tasks', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    stage = protocol['stage_c_music']
    expected = protocol['baseline_sha256']['saet_repro/data.py']
    if sha256(ROOT / 'saet_repro/data.py') != expected:
        raise ValueError('Baseline data builder changed; create a new protocol version')
    output = ROOT / stage['data_root']
    for year in stage['origins']:
        for offset in stage['offset_days']:
            print(output / f'cds_{year}_d{offset}.npz', flush=True)
    if args.list_tasks:
        return
    identity = {'protocol_sha256': sha256(args.protocol), 'data_builder_sha256': expected,
                'runner_sha256': sha256(Path(__file__)),
                'raw_sha256': {name: sha256(ROOT / 'data/raw' / name) for name in
                    ('CDs_and_Vinyl.csv', 'Digital_Music.csv', 'meta_CDs_and_Vinyl.jsonl.gz', 'meta_Digital_Music.jsonl.gz')},
                'origins': stage['origins'], 'offset_days': stage['offset_days']}
    frozen = output.parent / 'data_preparation.frozen.json'
    if frozen.exists():
        previous = json.loads(frozen.read_text())
        if previous['identity'] != identity:
            raise ValueError('Data preparation identity changed; use isolated new version')
    else:
        write_json(frozen, {'identity': identity, 'prepared_after_protocol_freeze_utc': datetime.now(timezone.utc).isoformat(),
            'visibility_note': 'Data preparation processes query labels mechanically after protocol freeze; no final outcome is used to select origins/configurations. Separate final-evaluation gate records access by model runners.'})
    command = [sys.executable, str(ROOT / 'scripts/prepare_data.py'), '--output-dir', str(output),
               '--cache-dir', str(output.parent / 'title_cache'), '--years', *map(str, stage['origins']),
               '--offsets', *map(str, stage['offset_days'])]
    subprocess.run(command, cwd=ROOT, check=True)
    manifests = [output / f'cds_{y}_d{d}.json' for y in stage['origins'] for d in stage['offset_days']]
    write_json(output.parent / 'data_preparation.complete.json', {'identity': identity,
        'manifest_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in manifests},
        'prefix_count': len(manifests), 'completed_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
