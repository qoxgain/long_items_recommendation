#!/usr/bin/env python3
"""Freeze the declared extension design without reading any new final query."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saet_repro.io import sha256, write_json


def main():
    path = ROOT / 'configs/extensions_v1.json'
    if path.exists():
        print(f'Existing immutable protocol: {path} SHA256={sha256(path)}')
        return
    selected = {}
    for method in ('activity_mean', 'activity_both'):
        p = ROOT / 'results/targets/cds_2015_d30' / method / 'selected.json'
        record = json.loads(p.read_text())
        selected[method] = {'config': record['config'], 'epochs': 6,
                            'selection_sha256': sha256(p), 'selection_prefix': 'cds_2015_d30'}
    baseline_files = sorted(ROOT.glob('saet_repro/*.py'))
    baseline_files = [p for p in baseline_files if not p.name.startswith('extension')]
    baseline_files += [ROOT / 'scripts/summarize_results.py', ROOT / 'requirements.txt',
                       ROOT / 'reports/artifact_manifest.json', ROOT / 'reports/REPRODUCTION_REPORT.md']
    protocol = {
        'version': 'extensions_v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'repository': 'qoxgain/long_tail_items_recommendation',
        'baseline_commit': 'cea4f4fa830fa3ee87a2af0041711951b5388be4',
        'starting_head': '832730b6eba97c91a795978a6366ca2ffe973026',
        'baseline_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in baseline_files},
        'input_manifest_sha256': sha256(ROOT / 'docs/extension_input_manifest.json'),
        'primary_endpoint': 'tail80 macro eligible-item NDCG@20',
        'aggregation': 'paired seed means within origin, equal origin means',
        'seeds': [11, 22, 33, 44, 55], 'bootstrap_resamples': 2000, 'bootstrap_seed': 2026,
        'bootstrap_scope': 'descriptive; stratified items and cross-origin unique-item clusters; shared-user dependence remains',
        'stage_a': {'status': 'authorized', 'origins': [2015, 2016, 2017], 'visibility': 'previously viewed development results',
                    'new_pairs': [['activity_both', 'activity'], ['activity_both', 'activity_decay'], ['activity_decay', 'activity']],
                    'retained_pairs': [['activity_both', 'activity_mean'], ['activity_decay', 'activity_mean']],
                    'no_training': True, 'output': 'reports/extensions_v1/stage_a'},
        'stage_b': {'status': 'authorized', 'visibility': 'exploratory development analysis',
                    'methods': ['mf_bpr', 'lightgcn', 'simplex', 'lightgcn_user', 'activity_mean', 'activity_both'],
                    'hpo_prefix': 'cds_2015_d30', 'grid': 'unchanged twelve configurations per baseline family; aligned LightGCN uses native LightGCN grid',
                    'hpo_seeds': [101, 202, 303], 'checkpoint_epochs': [6, 20, 50],
                    'selection': 'maximum mean across three HPO seeds; ties prefer fewer epochs then earlier grid index',
                    'evaluation_frequency': 'each historical epoch; selection only at declared checkpoints',
                    'extension_rule': {'initial_max_epoch': 50, 'extended_max_epoch': 100,
                        'trigger': 'family winner at epoch50 and improvement over same configuration epoch20 meets threshold',
                        'relative_threshold': 0.01, 'near_zero_denominator': 1e-8,
                        'absolute_threshold_when_near_zero': 1e-4, 'extend_all_family_candidates': True},
                    'hpo_trajectories': 216, 'final_trajectories': 90, 'final_origins': [2015, 2016, 2017],
                    'root': 'results_extensions/v1/stage_b'},
        'stage_c_music': {'status': 'authorized', 'origins': [2018, 2019, 2020],
                    'time_zone': 'UTC', 'origin_month_day': '01-01', 'fit_days': 365, 'query_days': 30,
                    'offset_days': [270, 180, 90, 60, 30, 0],
                    'visibility_at_freeze': 'not previously evaluated in the available workspace or supplied conversation; outside-workspace prior access is unknown',
                    'interpretation': 'prospective within this workflow; same-domain new-time validation, not independent-domain confirmation',
                    'base_selection': selected, 'historical_target_recipe_matches_deployment': True,
                    'policies': ['zero', 'uniform', 'semantic', 'full'],
                    'deployment_selection': 'first origin 2018 d30 only, independently per base, then frozen for 2019/2020',
                    'radii': [0, 0.01, 0.05, 0.1], 'temperatures': [0.1, 1, 10],
                    'ridge_alphas': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'primary_transfer_pair': ['activity_both/full', 'activity_both/zero'],
                    'primary_target_pair': ['activity_both/zero', 'activity'],
                    'secondary_pairs': [['activity_both/full', 'activity_both/uniform'], ['activity_both/full', 'activity_both/semantic'], ['activity_both/full', 'activity_mean/full']],
                    'final_policy_evaluations': 120, 'deterministic_activity_evaluations': 3,
                    'data_root': 'data_extensions/v1/music', 'root': 'results_extensions/v1/stage_c_music',
                    'gate': 'write final_query_opened.json with protocol and selection hashes before first final query evaluation'},
        'stage_c_new_domain': {'status': 'resource-gated; not included in current mandatory completion matrix',
                    'source': 'Movies_and_TV', 'target': 'Books', 'proposed_origins': [2018, 2019, 2020],
                    'feasibility': 'fit-only before any final query; minimum6000 eligible target users and16 source items',
                    'early_selection_offset_days': 330,
                    'head_request_archive_bytes': {'Books_reviews': 6216820644, 'Books_metadata': 4942125770,
                                                'Movies_and_TV_reviews': 2396672075, 'Movies_and_TV_metadata': 271381871},
                    'resource_note': 'About13.8GB compressed inputs, excluding projections, title caches and training states; A/B/C1 prioritized on32GB filesystem. Reassess after pilot; do not claim this domain evaluated.'},
        'execution': {'threads_per_worker': 1, 'maximum_training_workers': 3,
                    'old_artifacts': 'read-only reuse after validation',
                    'new_code_identity': 'each stage freezes finalized module/runner/dependency hashes before execution',
                    'checkpoint_reuse': 'verify identity and every required artifact hash; mismatch requires explicit recomputation in isolated outputs',
                    'failure_policy': 'record failed/missing cells; strict summaries cannot label partial matrices complete'}
    }
    write_json(path, protocol)
    path.with_suffix('.sha256').write_text(sha256(path) + '  extensions_v1.json\n')
    print(f'Frozen {path} SHA256={sha256(path)}')


if __name__ == '__main__':
    main()
