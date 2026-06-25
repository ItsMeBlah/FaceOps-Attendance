from __future__ import annotations

__all__ = ["ResultLogger"]


def __getattr__(name: str):
    if name == "ResultLogger":
        from app.logging.logger import ResultLogger

        return ResultLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
