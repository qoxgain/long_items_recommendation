"""New transfer interfaces: temporal gate, cache identity and stronger targets."""
from datetime import datetime, timezone
import json
import os

import numpy as np
import pytest

from saet_repro import extension_transfer as transfer
from saet_repro.data import save_data
from saet_repro.evaluation import evaluate_many
from saet_repro.io import write_json, sha256
from saet_repro.target import Model, FeatureStore
from saet_repro.utility import RouteContext
from test_utility import tiny_data, constant_ridges


def protocol():
    return json.loads((transfer.ROOT / 'configs/extensions_v1.json').read_text())


def save_prefix(runner, year, offset, data=None):
    data = tiny_data() if data is None else data
    end = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000) - offset * transfer.DAY
    shift = end - int(data['fit_end'])
    for domain in ('target', 'source'):
        for field in ('event_times', 'all_rating_times'):
            data[f'{domain}_{field}'] = data[f'{domain}_{field}'] + shift
    data.update(fit_start=np.array(end - 365 * transfer.DAY), fit_end=np.array(end),
                query_end=np.array(end + 30 * transfer.DAY),
                cutoffs=np.asarray([end + d * transfer.DAY for d in (0, 30, 60, 90)]))
    path = runner.path(year, offset)
    save_data(path, data)
    write_json(path.with_suffix('.json'), {
        'prefix': path.stem, 'origin_year': year, 'offset_days': offset,
        'fit_start': int(data['fit_start']), 'fit_end': end, 'query_end': int(data['query_end']),
        'data_sha256': sha256(path), 'code_sha256': sha256(transfer.ROOT / 'saet_repro/data.py')})
    return data


def make_runner(tmp_path):
    path = tmp_path / 'protocol.json'
    write_json(path, protocol())
    runner = transfer.Runner(tmp_path / 'data', tmp_path / 'results', path)
    runner.freeze()
    return runner


def test_frozen_grid_and_zero_first_tie_rule():
    configs = transfer.hpo_configurations()
    assert len(configs) == 37
    rows = [{'name': name, 'configuration': config, 'score': .1} for name, config in configs.items()]
    selected = transfer.choose_policies(rows)
    assert len(selected) == 4
    assert all(config['radius'] == 0 for config in selected.values())
    assert selected['full']['temperature'] == .1
    rows[-1]['score'] = .11
    assert transfer.choose_policies(rows)['full'] == rows[-1]['configuration']
    with pytest.raises(ValueError, match='all 37'):
        transfer.choose_policies(rows[:-1])
    rows[-1]['configuration'] = dict(rows[-1]['configuration'], radius=9)
    with pytest.raises(ValueError, match='differs'):
        transfer.choose_policies(rows)


def test_protocol_mismatch_and_query_firewall(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    calls = []
    monkeypatch.setattr(transfer, 'load_data', lambda path: calls.append(path))
    with pytest.raises(RuntimeError, match='Final queries require'):
        runner.load_prefix(2018, 0)
    with pytest.raises(RuntimeError, match='missing'):
        runner.open_final_queries()
    assert calls == [] and not runner.gate.exists()
    changed = protocol()
    changed['stage_c_music']['base_selection']['activity_mean']['config']['lr'] = .001
    write_json(runner.protocol_path, changed)
    with pytest.raises(ValueError, match='configuration mismatch'):
        runner.freeze()
    assert not runner.gate.exists()


def test_temporal_manifest_archive_and_source_index_checks(tmp_path):
    runner = make_runner(tmp_path)
    data = save_prefix(runner, 2018, 60)
    runner.load_prefix(2018, 60)
    path = runner.path(2018, 60)
    manifest = json.loads(path.with_suffix('.json').read_text())
    write_json(path.with_suffix('.json'), {**manifest, 'fit_end': manifest['fit_end'] + transfer.DAY})
    with pytest.raises(RuntimeError, match='temporal/build'):
        runner.load_prefix(2018, 60)
    write_json(path.with_suffix('.json'), manifest)
    bad = dict(data, fit_end=np.array(int(data['fit_end']) + transfer.DAY))
    with pytest.raises(RuntimeError, match='archive temporal'):
        runner.validate_data(bad, 2018, 60)
    bad = dict(data, source_event_users=data['source_event_users'].copy())
    bad['source_event_users'][0] = len(data['source_user_ids'])
    with pytest.raises(RuntimeError, match='Event bounds invalid: source'):
        runner.validate_data(bad, 2018, 60)
    bad = dict(data, target_all_rating_times=data['target_all_rating_times'].copy())
    bad['target_all_rating_times'][0] = int(data['fit_end'])
    with pytest.raises(RuntimeError, match='Activity history crosses'):
        runner.validate_data(bad, 2018, 60)
    with path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(RuntimeError, match='differs from manifest'):
        runner.load_prefix(2018, 60)
    assert not runner.gate.exists()


def test_freeze_enforces_original_selection_and_calendar(tmp_path):
    runner = make_runner(tmp_path)
    for key, value in (('fit_days', 364), ('query_days', 31), ('time_zone', 'local')):
        changed = protocol()
        changed['stage_c_music'][key] = value
        write_json(runner.protocol_path, changed)
        with pytest.raises(ValueError, match=key):
            runner.freeze()
    changed = protocol()
    changed['stage_c_music']['base_selection']['activity_mean']['selection_sha256'] = '0' * 64
    write_json(runner.protocol_path, changed)
    with pytest.raises(ValueError, match='Old target selection'):
        runner.freeze()


def test_strict_cache_quarantines_changed_bytes_and_blocks_reselection(tmp_path):
    runner = make_runner(tmp_path)
    directory = runner.result_dir / 'cache'
    artifact, marker = directory / 'test.npz', directory / 'complete.json'
    transfer.save_npz(artifact, x=np.arange(3))
    frozen = {**runner.identity, 'config': 'one'}
    runner.finish(marker, frozen, {'test': artifact})
    assert runner.cached(marker, frozen, {'test': artifact})
    with artifact.open('ab') as handle:
        handle.write(b'changed')
    assert runner.cached(marker, frozen, {'test': artifact}) is None
    assert list((directory / 'stale').glob('*/reason.json'))
    assert not marker.exists() and not artifact.exists()
    transfer.save_npz(artifact, x=np.arange(3))
    runner.finish(marker, frozen, {'test': artifact})
    write_json(runner.gate, {'identity': runner.identity})
    with artifact.open('ab') as handle:
        handle.write(b'changed again')
    with pytest.raises(RuntimeError, match='Selection dependency invalid'):
        runner.cached(marker, frozen, {'test': artifact})
    assert marker.exists()


@pytest.mark.parametrize('base', list(transfer.BASE_CONFIGS))
def test_both_strong_bases_preserve_exact_null_scores_ranks_metrics(base):
    data = tiny_data()
    config = transfer.BASE_CONFIGS[base]
    model = Model(config, FeatureStore(data, config))
    before = model.theta.copy(), model.beta.copy(), model.alpha
    context = RouteContext(data, model, source_epochs=0)
    configs = {p: dict(policy=p, radius=0., temperature=.1) for p in transfer.POLICIES}
    scorer = transfer.CheckedScorer(context, constant_ridges(.1), configs)
    results = evaluate_many(data, 1, scorer, scorer.names, batch_size=1)
    assert scorer.scored_items > 0
    assert len(set(scorer.hashes().values())) == 1
    for policy in transfer.POLICIES:
        for name in results['canonical']:
            np.testing.assert_array_equal(results[policy][name], results['canonical'][name])
    np.testing.assert_array_equal(model.theta, before[0])
    np.testing.assert_array_equal(model.beta, before[1])
    assert model.alpha == before[2]


def test_historical_same_configuration_no_disk_banks_and_recomputes_stale_data(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    path = runner.path(2018, 60)
    save_prefix(runner, 2018, 60)
    unused = tmp_path / 'forbidden_disk_bank'
    monkeypatch.setenv('SAET_FEATURE_CACHE', str(unused))
    original = transfer.train_target
    observed = []

    def check_training(data, config, seed, **kwargs):
        assert 'SAET_FEATURE_CACHE' not in os.environ
        observed.append(config)
        return original(data, config, seed, **kwargs)

    monkeypatch.setattr(transfer, 'train_target', check_training)
    labels = runner.historical_fold('activity_mean', 2018, 60)
    assert len(labels['labels'])
    assert observed == [transfer.BASE_CONFIGS['activity_mean']]
    assert observed[0].lr == .01
    runner.historical_fold('activity_mean', 2018, 60)
    assert len(observed) == 1
    changed = tiny_data()
    changed['target_x'][0, 0] += .01
    save_prefix(runner, 2018, 60, changed)
    runner.historical_fold('activity_mean', 2018, 60)
    assert len(observed) == 2
    assert os.environ['SAET_FEATURE_CACHE'] == str(unused)
    assert not unused.exists()
    assert not runner.gate.exists()
    with pytest.raises(ValueError, match='cannot load'):
        runner.historical_fold('activity_mean', 2018, 0)


def test_complete_tiny_selection_gate_and_frozen_final(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    for year in transfer.ORIGINS:
        for offset in (*transfer.OFFSETS, 30, 0):
            save_prefix(runner, year, offset)
    real_load = transfer.load_data
    opened = []

    def guarded_load(path):
        if path.stem.endswith('_d0'):
            assert runner.gate.exists()
            opened.append(path)
        return real_load(path)

    monkeypatch.setattr(transfer, 'load_data', guarded_load)
    ridges = {(base, year): runner.fit_ridges(base, year)
              for base in transfer.BASE_CONFIGS for year in transfer.ORIGINS}
    selections = {base: runner.search(base, ridges[base, 2018]) for base in transfer.BASE_CONFIGS}
    assert not opened and not runner.gate.exists()
    # A self-consistent marker is insufficient: every field must match its cell.
    directory = runner.historical_directory('activity_mean', 2018, 60)
    for marker_name, key, value in (
            ('training.complete.json', 'seed', 202),
            ('training.complete.json', 'epochs', 50),
            ('source_activity.json', 'source_positive_cap', 5),
            ('result.json', 'role', 'ridge_training')):
        marker = directory / marker_name
        original = marker.read_text()
        record = json.loads(original)
        record['frozen'][key] = value
        write_json(marker, record)
        with pytest.raises(RuntimeError, match='dependency mismatch'):
            runner.open_final_queries()
        assert not opened and not runner.gate.exists()
        marker.write_text(original)
    marker = runner.hpo_directory('activity_mean') / 'selected.json'
    original = marker.read_text()
    record = json.loads(original)
    record['trials'][-1]['score'] = 1.
    record['policies'] = transfer.choose_policies(record['trials'])
    write_json(marker, record)
    with pytest.raises(RuntimeError, match='trial records differ'):
        runner.open_final_queries()
    marker.write_text(original)
    assert not opened and not runner.gate.exists()
    loaded_ridges, loaded_selections = runner.open_final_queries()
    assert loaded_selections == selections and len(loaded_ridges) == 6
    gate = json.loads(runner.gate.read_text())
    assert gate['protocol_sha256'] == sha256(runner.protocol_path)
    assert len(gate['selection_dependencies']) == 84
    first_gate_hash = sha256(runner.gate)
    runner.open_final_queries()
    assert sha256(runner.gate) == first_gate_hash
    runner.final_evaluation('activity_both', 2018, 11, loaded_ridges['activity_both', 2018], selections['activity_both'])
    assert len(opened) == 1
    directory = runner.result_dir / 'activity_both/final/cds_2018_d0/seed_11'
    completed = json.loads((directory / 'complete.json').read_text())
    assert completed['null_check']['scores_exact']
    assert completed['null_check']['all_rank_and_metric_arrays_exact']
    with pytest.raises(RuntimeError, match='checked final-prefix loader'):
        runner.activity_evaluation(2018, tiny_data())
    data = runner._final_data[1]
    wrong_ridges = dict(loaded_ridges['activity_both', 2018])
    from copy import deepcopy
    wrong_ridges = deepcopy(wrong_ridges)
    wrong_ridges['full'].coefficient[0] += 1
    with pytest.raises(RuntimeError, match='Deployment ridges differ'):
        runner.final_evaluation('activity_both', 2018, 11, wrong_ridges, selections['activity_both'], data)
    runner.activity_evaluation(2018)
    assert (runner.result_dir / 'activity/final/cds_2018_d0/seed_0/test.npz').exists()
    with (runner.hpo_directory('activity_mean') / 'search/full_11.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(RuntimeError, match='Selection dependency invalid'):
        runner.open_final_queries()
    assert sha256(runner.gate) == first_gate_hash


def test_protocol_cannot_change_after_query_opening(tmp_path):
    runner = make_runner(tmp_path)
    write_json(runner.gate, {'identity': runner.identity})
    changed = protocol()
    changed['explanation'] = 'changed after opening'
    write_json(runner.protocol_path, changed)
    with pytest.raises(RuntimeError, match='after final queries opened'):
        runner.freeze()
