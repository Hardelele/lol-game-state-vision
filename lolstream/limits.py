"""Скромный режим тяжёлого офлайн-прогона: пониженный приоритет и мало потоков.

Прогон набора идёт на той же машине, где владелец играет и где работают
copilot, OBS и другие прогоны. Без ограничений torch, numpy (BLAS), OpenCV и
ffmpeg забирают все ядра, и машина перестаёт отзываться (ЦП 98%). Поэтому
по умолчанию: приоритет процесса BelowNormal (его наследуют ffmpeg и
yt-dlp), не больше `threads` потоков у torch, BLAS, OpenMP и OpenCV.

`env_limits` — до импорта numpy и torch (переменные окружения читаются при
загрузке библиотек), `apply_limits` — после: число потоков torch и OpenCV и
приоритет процесса.
"""

from __future__ import annotations

import os
import sys

DEFAULT_THREADS = 4
ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
       "VECLIB_MAXIMUM_THREADS")
BELOW_NORMAL = 0x00004000       # BELOW_NORMAL_PRIORITY_CLASS
IDLE = 0x00000040               # IDLE_PRIORITY_CLASS


def env_limits(threads: int = DEFAULT_THREADS) -> None:
    """Потоки BLAS / OpenMP — через окружение; уже заданное не трогается."""
    for k in ENV:
        os.environ.setdefault(k, str(threads))


def lower_priority(idle: bool = False) -> str:
    """Приоритет своего процесса ниже обычного (дети его наследуют)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Без типов дескриптор (-1, «свой процесс») уходит 32-битным int и
        # на 64-битной системе не узнаётся: приоритет молча не менялся.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k32.SetPriorityClass.restype = wintypes.BOOL
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), IDLE if idle else BELOW_NORMAL)
        return ("Idle" if idle else "BelowNormal") if ok else "не изменён"
    try:
        os.nice(19 if idle else 10)
        return f"nice {os.nice(0)}"
    except OSError:
        return "не изменён"


def apply_limits(threads: int = DEFAULT_THREADS, priority: bool = True) -> dict:
    """Потоки torch и OpenCV и приоритет — после импорта библиотек."""
    out = {"threads": threads}
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except (ImportError, RuntimeError):
        pass
    try:
        import cv2
        cv2.setNumThreads(threads)
    except ImportError:
        pass
    out["priority"] = lower_priority() if priority else "обычный"
    return out
