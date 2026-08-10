"""Probe registry and loading.

Adding a test can be done three ways, in ascending order of effort:

1.  Drop a `.py` file in this package. It is discovered on the next run.
2.  Point `probe_dirs` (config) or `--probe-dir` (flag) at a folder of your own
    `.py` files. Same `@register` decorator, no need to touch the install.
3.  Write a `custom_probes` block in `config.yaml` and skip Python entirely,
    for the common case of "send these prompts, fail if the reply does / does
    not contain X".

Nothing imports probes by name, so the registry is whatever has been registered
by the time `all_probes()` is called.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import Any

from ..protocol import Probe

_REGISTRY: dict[str, type[Probe]] = {}


def register(cls: type[Probe]) -> type[Probe]:
    """Class decorator that adds a probe to the registry.

    Raises on a missing or duplicate id, so a copy-paste mistake fails loudly at
    import time rather than silently shadowing another probe.
    """
    if not cls.id:
        raise ValueError(f"{cls.__name__} needs an id")
    if not cls.owasp:
        raise ValueError(f"{cls.__name__} needs an owasp category (e.g. LLM10:2025 Unbounded Consumption)")
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate probe id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def _discover() -> None:
    """Import every non-underscore module in this package, registering its probes."""
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")


def load_probe_dirs(dirs: list[Path]) -> list[str]:
    """Import every `.py` file under each directory so its probes register.

    Returns a list of human-readable notes about what was loaded or skipped, for
    the CLI to print. Import errors are reported rather than raised, so one bad
    file in a directory does not sink the whole run.
    """
    notes: list[str] = []
    for directory in dirs:
        if not directory.exists():
            notes.append(f"probe dir not found: {directory}")
            continue
        for file in sorted(directory.glob("*.py")):
            if file.name.startswith("_"):
                continue
            module_name = f"lmcorral_userprobes_{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    notes.append(f"could not load {file}")
                    continue
                module = importlib.util.module_from_spec(spec)
                # Register before executing so a probe module that imports itself
                # by name during definition still resolves.
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                notes.append(f"loaded {file}")
            except Exception as exc:  # noqa: BLE001 - a bad user file must not crash the run
                notes.append(f"error loading {file}: {type(exc).__name__}: {exc}")
    return notes


def load_declarative(specs: list[dict[str, Any]]) -> list[str]:
    """Turn `custom_probes` config blocks into registered probe classes.

    Kept import-light on purpose; the actual construction lives in
    `declarative.py` so this module has no heavy dependencies.
    """
    if not specs:
        return []
    # Deferred: importing this at module top would pull in the probe modules
    # (via safety) before `register` below is defined, a circular import.
    from .declarative import build_declarative_probe

    notes: list[str] = []
    for spec in specs:
        try:
            cls = build_declarative_probe(spec)
            register(cls)
            notes.append(f"registered custom probe {cls.id!r}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            name = spec.get("id", "<unnamed>")
            notes.append(f"error in custom probe {name!r}: {type(exc).__name__}: {exc}")
    return notes


def all_probes() -> dict[str, type[Probe]]:
    """Every registered probe, built-ins included, sorted by id."""
    _discover()
    return dict(sorted(_REGISTRY.items()))


def select(patterns: list[str] | None) -> list[Probe]:
    """Instantiate probes whose id equals or starts with any of `patterns`.

    An empty or missing pattern list selects everything. A pattern that matches
    nothing raises, so a typo in `--probe` fails fast instead of quietly running
    zero probes.
    """
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
