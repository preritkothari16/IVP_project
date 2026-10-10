"""Cap BLAS/OpenMP thread counts BEFORE torch is imported.

This machine has 16 logical cores and ~24 GB of RAM, and the default behaviour of the bundled
OpenBLAS is to spawn one worker per core. With several hundred processes already resident that
exhausts the commit limit, and two distinct failures result:

  OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
  OSError: [WinError 1455] The paging file is too small for this operation to complete.
      ... torch/lib/curand64_10.dll

The second is the nastier one because it looks like a broken install rather than memory pressure.
Both go away when the thread pools are pinned to a small number.

Import this module FIRST in any entry point that imports torch. Setting the variables after
``import torch`` has no effect, because the pools are sized at library load time.

Override with the environment if a different machine needs it::

    STRAWBERRY_THREADS=8 python ...
"""

import os

# A single-threaded BLAS is enough here: the workload is GPU-bound at batch 32, and the dataloader
# workers handle image decoding. More BLAS threads buy nothing and cost the whole run its ability
# to start.
_DEFAULT = "2"


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


_threads = _env("STRAWBERRY_THREADS") or _DEFAULT
# 0 means "leave whatever torch decides" for intra-op parallelism.
_intra = _env("STRAWBERRY_TORCH_THREADS") or _threads

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, _threads)
    # Respect an explicit caller override rather than forcing.
    if _env(_var):
        os.environ[_var] = _env(_var)

os.environ.setdefault("STRAWBERRY_THREADS_APPLIED", _threads)

try:  # only takes effect if torch is imported afterwards
    import torch  # noqa: E402,F401

    torch.set_num_threads(max(1, int(_intra)))
except Exception:  # pragma: no cover - torch may be absent in a data-only context
    pass