"""Safety and completeness checks for the execution-only CF scheduler."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parallel_cf_test", ROOT / "scripts/run_extension_cf_parallel.py")
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def test_jobs_preserve_full_unique_original_matrix_and_extension():
    for method in parallel.original.METHODS:
        initial = parallel.historical_jobs(method, 50)
        extended = parallel.historical_jobs(method, 100)
        assert len(initial) == len(set(initial)) == 36
        assert [job[:3] for job in initial] == [job[:3] for job in extended]
        assert {job[1] for job in initial} == set(range(12))
        assert {job[2] for job in initial} == {101, 202, 303}
        final = parallel.final_jobs(method)
        assert len(final) == len(set(final)) == 15
        assert {job[1] for job in final} == {2015, 2016, 2017}
        assert {job[2] for job in final} == {11, 22, 33, 44, 55}


def test_live_legacy_writer_detection_ignores_zombies_and_other_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, cwd, state in ((1001, checkout, "S"), (1002, checkout, "Z"), (1003, other, "S")):
        directory = proc / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"python\0scripts/run_extension_cf.py\0--methods\0lightgcn_user\0")
        (directory / "status").write_text(f"State:\t{state} (state)\n")
        (directory / "cwd").symlink_to(cwd)
    assert parallel.legacy_writers(checkout, proc) == [1001]


def test_main_refuses_live_legacy_before_any_protocol_or_data_read(monkeypatch):
    monkeypatch.setattr(parallel, "legacy_writers", lambda: [13701])
    monkeypatch.setattr(parallel.original, "validate_global_protocol",
                        lambda path: pytest.fail("Protocol should not be read after active-writer rejection"))
    with pytest.raises(SystemExit, match="13701"):
        parallel.main(["--method", "lightgcn_user"])


def test_changed_input_rejected_before_decode_even_if_data_cached(tmp_path, monkeypatch):
    path = tmp_path / "cds_2015_d30.npz"
    path.write_bytes(b"frozen input")
    digest = parallel.sha256(path)
    monkeypatch.setattr(parallel, "_CONTEXT", {"data_sha256": {path.name: digest}})
    monkeypatch.setattr(parallel, "_DATA_PATH", path)
    monkeypatch.setattr(parallel, "_DATA", {"cached": True})
    assert parallel._load(path) == {"cached": True}
    path.write_bytes(b"changed input")
    monkeypatch.setattr(parallel.original, "load_data",
                        lambda path: pytest.fail("Changed input must not be decoded"))
    with pytest.raises(ValueError, match="Input changed"):
        parallel._load(path)


def test_final_selection_failure_precedes_final_data_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "_CONTEXT", {
        "results_dir": str(tmp_path), "data_dir": str(tmp_path),
        "identity": {}, "selected_sha256": "unused", "retry_failed": False})
    def reject(*args):
        raise ValueError("invalid selection")
    monkeypatch.setattr(parallel.original, "validate_selection", reject)
    monkeypatch.setattr(parallel.original, "preserve_failure", lambda *args: None)
    monkeypatch.setattr(parallel, "_load",
                        lambda path: pytest.fail("Final data cannot open before selection validation"))
    with pytest.raises(ValueError, match="invalid selection"):
        parallel._final(("lightgcn_user", 2015, 11))
