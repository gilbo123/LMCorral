"""Probe registry.

Adding a test is one file and one decorator. Nothing imports probes by name, so
a new module dropped in this package is discovered on the next run.
"""

from __future__ import annotations

import importlib
import pkgutil

from ..protocol import Probe

_REGISTRY: dict[str, type[Probe]] = {}


def register(cls: type[Probe]) -> type[Probe]:
    if not cls.id:
        raise ValueError(f"{cls.__name__} needs an id")
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate probe id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def _discover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")


def all_probes() -> dict[str, type[Probe]]:
    _discover()
    return dict(sorted(_REGISTRY.items()))


def select(patterns: list[str] | None) -> list[Probe]:
    """Instantiate probes whose id starts with any of `patterns`."""
    registry = all_probes()
    if not patterns:
        return [cls() for cls in registry.values()]
    chosen: list[Probe] = []
    for pattern in patterns:
        matched = [
            cls for pid, cls in registry.items() if pid == pattern or pid.startswith(pattern)
        ]
        if not matched:
            raise KeyError(f"no probe matching {pattern!r}; try `lmcorral probes`")
        chosen.extend(cls() for cls in matched)
    seen: set[str] = set()
    unique = []
    for probe in chosen:
        if probe.id not in seen:
            seen.add(probe.id)
            unique.append(probe)
    return unique
