"""Tests for the in-process reproducibility switches (viking_splatad.splatad.determinism).

The Mac test venv has no torch or nerfstudio, so stand-in modules are placed in ``sys.modules``.
"""

import concurrent.futures
import sys
import threading
import types

from viking_splatad.splatad import determinism


def test_deterministic_mode_requested_only_for_exact_one():
    """Only VIKING_DETERMINISTIC=1 turns the switches on."""
    assert determinism.deterministic_mode_requested({determinism.DETERMINISTIC_ENV_VAR: "1"})
    assert not determinism.deterministic_mode_requested({})
    assert not determinism.deterministic_mode_requested({determinism.DETERMINISTIC_ENV_VAR: "0"})
    assert not determinism.deterministic_mode_requested({determinism.DETERMINISTIC_ENV_VAR: "true"})


def test_single_worker_executor_overrides_requested_workers_and_keeps_order():
    """max_workers=2 is overridden to 1: one thread runs every task, in submission order."""
    task_count = 20
    thread_names_by_task = {}

    def record_thread(task_index):
        thread_names_by_task[task_index] = threading.current_thread().name
        return task_index

    # The caller asks for 2 workers (neurad's value); the stand-in must still use one thread.
    with determinism.single_worker_executor(max_workers=2) as executor:
        results = list(executor.map(record_thread, range(task_count)))
    assert results == list(range(task_count))
    assert len(set(thread_names_by_task.values())) == 1


def test_enable_deterministic_mode_patches_pool_and_torch_flags(monkeypatch):
    """The lidar module's pool is replaced and the three torch switches are set as documented."""
    recorded_calls = []
    # Stand-in torch with the attributes enable_deterministic_mode touches; benchmark starts True
    # like nerfstudio's train.py leaves it.
    fake_cudnn = types.SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(cudnn=fake_cudnn)
    fake_torch.use_deterministic_algorithms = lambda mode, warn_only=False: recorded_calls.append((mode, warn_only))
    fake_lidar_module = types.ModuleType(determinism.LIDAR_DATAMANAGER_MODULE)
    fake_lidar_module.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, determinism.LIDAR_DATAMANAGER_MODULE, fake_lidar_module)

    determinism.enable_deterministic_mode()
    # Idempotent: a second call must leave the same state.
    determinism.enable_deterministic_mode()

    assert fake_lidar_module.ThreadPoolExecutor is determinism.single_worker_executor
    assert fake_cudnn.benchmark is False
    assert fake_cudnn.deterministic is True
    assert recorded_calls == [(True, True), (True, True)]


def test_sbatch_exports_determinism_variables_before_training():
    """The training command is preceded by the three exports that must exist before Python starts."""
    from pathlib import Path

    sbatch_text = (Path(__file__).resolve().parents[2] / "scripts/slurm/splatad_run.sbatch").read_text()
    training_position = sbatch_text.index("\nns-train splatad")
    export_position = sbatch_text.index("export VIKING_DETERMINISTIC=1")
    assert export_position < training_position
    export_line = sbatch_text[export_position:training_position]
    assert 'PYTHONHASHSEED="$TRAIN_SEED"' in export_line
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in export_line
