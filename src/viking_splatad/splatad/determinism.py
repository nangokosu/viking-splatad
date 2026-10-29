"""Run-to-run reproducibility switches for SplatAD training (applied in-process, no env rebuild).

Same seed alone does not reproduce a run (see ``scripts/slurm/splatad_run.sbatch``). Two sources
are removable and are switched off here; the third is not:

1. **RNG race (removed).** neurad's lidar cache draws ``torch.randperm`` from the one global
   generator inside a 2-thread pool (``full_images_lidar_datamanager.py``), so which sweep gets
   which permutation depends on thread scheduling and every later draw (Gaussian init, sampling
   order, MCMC noise) shifts. One worker runs the sweeps in submission order, so the stream is a
   function of the seed only.
2. **cuDNN / PyTorch kernel choice (removed).** ``cudnn.benchmark`` autotunes by timing, and some
   PyTorch ops pick a non-deterministic kernel. ``cudnn.deterministic`` fixes the cuDNN choice.
   ``torch.use_deterministic_algorithms`` is deliberately NOT called: it routes gsplat's
   ``v[sampled_idxs] = 0`` (MCMC relocation) through PyTorch's deterministic ``index_put_``, which
   hits an INTERNAL ASSERT in ``Indexing.cu`` and killed every smoke run at step ~590 (jobs
   224547/224548); gsplat's ``bincount``/``cumsum`` have no deterministic version anyway.
3. **CUDA float atomics (NOT removed).** The gsplat/SplatAD backward kernels and the tcnn lidar
   decoder add gradients with ``atomicAdd`` in a hardware-dependent order. Float addition is not
   associative, so runs drift at the last-bit level and MCMC/Adam amplify it. Only a kernel rewrite
   would fix that, so runs agree closely at first and then diverge slowly.

Everything is gated by ``VIKING_DETERMINISTIC=1`` (exported by the training step of
``splatad_run.sbatch``), so renders and tests are unaffected unless they opt in.
"""

import concurrent.futures
import importlib
import os
from typing import Dict

# Environment variable that turns the switches on; "1" = on.
DETERMINISTIC_ENV_VAR = "VIKING_DETERMINISTIC"

# neurad module whose lidar cache holds the racy thread pool, and the pool's name inside it.
LIDAR_DATAMANAGER_MODULE = "nerfstudio.data.datamanagers.full_images_lidar_datamanager"
EXECUTOR_ATTRIBUTE = "ThreadPoolExecutor"


def deterministic_mode_requested(environment: Dict[str, str] = None) -> bool:
    """Whether ``VIKING_DETERMINISTIC=1`` is set.

    Args:
        environment: Mapping to read (default ``os.environ``); a parameter so tests need no global state.

    Returns:
        True when the switches should be applied.
    """
    environment = os.environ if environment is None else environment
    return environment.get(DETERMINISTIC_ENV_VAR) == "1"


def single_worker_executor(*args, **kwargs) -> concurrent.futures.ThreadPoolExecutor:
    """Drop-in for ``ThreadPoolExecutor`` that always uses one worker.

    ``executor.map`` hands tasks to a single worker in submission order, so tasks that draw from
    a shared RNG draw in a fixed order.

    Args:
        *args: Forwarded to ``ThreadPoolExecutor``.
        **kwargs: Forwarded; any ``max_workers`` is overridden with 1.

    Returns:
        A one-thread ``ThreadPoolExecutor``.
    """
    # Override rather than reject: the caller asks for 2 workers, which is the race.
    kwargs["max_workers"] = 1
    return concurrent.futures.ThreadPoolExecutor(*args, **kwargs)


def enable_deterministic_mode() -> None:
    """Apply switches 1 and 2 of the module docstring (idempotent).

    Order matters only in that the lidar cache must not have started yet: the dataparser calls this
    before the datamanager caches anything, and after ``train_loop`` has seeded the RNGs.
    """
    # Imported here: the Mac test venv has no torch, and the module must import without it.
    import torch

    # Switch 1: the lidar cache's pool -> one worker (module attribute is looked up at call time).
    lidar_datamanager = importlib.import_module(LIDAR_DATAMANAGER_MODULE)
    setattr(lidar_datamanager, EXECUTOR_ATTRIBUTE, single_worker_executor)
    # Switch 2: fixed cuDNN kernel choice. train.py sets benchmark=True at import; undo it here.
    # (No torch.use_deterministic_algorithms: it crashes MCMC relocation, see the module docstring.)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
