"""Prevent idle sleep while a long job runs (Windows only; opt-in).

Why this exists: in training run3, PPO update 4 took 13,828 s of wall time
against 120-217 s for updates 1-3 and 5, with the same workload (14-19 k
decisions per update). The Windows power log records the system returning from
a low-power state at the moment update 4 finished. A multi-day training run on
this laptop would be dominated by sleep.

Scope, deliberately narrow: ``SetThreadExecutionState(ES_CONTINUOUS |
ES_SYSTEM_REQUIRED)`` asks Windows not to idle-sleep while the calling thread is
alive, and the request disappears when the process exits. No system power
setting is changed. It prevents IDLE sleep only -- closing the lid or choosing
Sleep still suspends the machine, and on other platforms this is a no-op.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(enabled: bool = True) -> Iterator[bool]:
    """Yield True if idle sleep is being held off, False otherwise."""
    active = False
    if enabled and sys.platform == "win32":
        import ctypes

        previous = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        active = previous != 0
    try:
        yield active
    finally:
        if active:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
