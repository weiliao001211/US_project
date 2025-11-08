from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


def _tty_available() -> bool:
    try:
        import sys

        return sys.stderr.isatty()
    except Exception:
        return False


@dataclass
class ProgressConfig:
    enabled: bool = True
    backend: str = "tqdm"  # or "none"
    leave: bool = False
    ncols: int = 100
    unit: str = ""
    position: int = 0


class Progress:
    """Backend-agnostic progress wrapper. No-ops when disabled/unavailable."""

    def __init__(self, cfg: Optional[ProgressConfig] = None):
        self.cfg = cfg or ProgressConfig()
        self._tqdm = None
        if self.cfg.enabled and _tty_available() and self.cfg.backend == "tqdm":
            try:
                from tqdm.auto import tqdm as _tqdm

                self._tqdm = _tqdm
            except Exception:
                self._tqdm = None

    def iter(
        self,
        it: Iterable,
        *,
        total: Optional[int] = None,
        desc: str = "",
        unit: Optional[str] = None,
    ) -> Iterator:
        if self._tqdm is None:
            return iter(it)
        return self._tqdm(
            it,
            total=total,
            desc=desc,
            unit=unit or self.cfg.unit,
            leave=self.cfg.leave,
            ncols=self.cfg.ncols,
            position=self.cfg.position,
        )

    @contextmanager
    def task(self, *, total: Optional[int], desc: str, unit: str = ""):
        """Manual progress for non-iterative phases; yields an update(n) fn."""
        if self._tqdm is None or total is None:
            yield lambda n=1: None
            return
        bar = self._tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=self.cfg.leave,
            ncols=self.cfg.ncols,
            position=self.cfg.position,
        )
        try:
            yield bar.update
        finally:
            bar.close()
