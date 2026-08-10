"""Configuration.

Optional `config.yaml` in the working directory (or ``--config``) holds limits,
reports, and probe settings. ``target.url`` and ``target.model`` can live there
or be passed on the command line as ``--target`` and ``--model`` — CLI wins
when both are set.

Optional CLI flags (``--probe``, ``--docx``, etc.) override non-target settings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml
from typing import Any

DEFAULT_CONFIG_NAMES = ("config.yaml", "config.yml")

#: `${VAR}` inside any config string is replaced from the environment. Secrets
#: belong in the environment, not in a file that ends up committed.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    """Raised for a malformed config file, with a message aimed at the user."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class Limits:
    """Tunables shared by the probes and the monitors they arm.

    These are the knobs worth touching. Budgets are deliberately generous: a
    token budget that trips at 64 chunks proves nothing about a model, only
    about the budget.
    """

    #: Stream chunks a generation may produce before it is cut off.
    token_budget: int = 1024 #multiple paragraphs
    #: Seconds a single generation may run before it is cut off.
    wall_clock_seconds: float = 120.0
    #: Seconds of silence between chunks before the stream is judged hung.
    token_gap_seconds: float = 20.0
    #: `num_predict`/`max_tokens` for probes that want a bounded answer. Probes
    #: testing for runaway output ignore this and request no ceiling at all.
    max_tokens: int = 4096 #multiple paragraphs
    #: Sampling temperature for probes that do not pin their own.
    temperature: float = 0.9

    #: Shortest repeating unit, in characters, treated as a cycle.
    repetition_min_period: int = 3
    #: Longest repeating unit to search for. Raising this costs time per check.
    repetition_max_period: int = 256 #multiple sentences
    #: Consecutive identical blocks that constitute a cycle.
    repetition_cycles: int = 5
    #: Identical whole lines that constitute a cycle.
    repetition_line_repeats: int = 8
    #: Check for repetition every N chunks rather than on every one.
    repetition_check_every: int = repetition_min_period * repetition_cycles

    def merged(self, overrides: dict[str, Any]) -> Limits:
        """Return a copy with `overrides` applied, validating key names."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(
                f"unknown limit(s): {', '.join(sorted(unknown))}. "
                f"Valid limits are: {', '.join(sorted(known))}"
            )
        return replace(self, **overrides)


@dataclass
class TargetConfig:
    """Where the model is and how to talk to it."""

    #: Base URL. A `/v1` in the path selects the OpenAI-compatible client.
    url: str = ""
    model: str = ""
    #: Bearer token for OpenAI-compatible endpoints. Use `${VAR}`, not a literal.
    api_key: str = ""
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 120.0


@dataclass
class ReportConfig:
    """Where results are written. Any format may be disabled with `null`."""

    jsonl: Path | None = Path("lmcorral-report.jsonl")
    docx: Path | None = None
    #: Cap on transcript text stored per turn, so a runaway probe does not
    #: produce a hundred-megabyte report.
    max_transcript_chars: int = 4000


@dataclass
class ProbeServerConfig:
    """Optional local HTTP sink for SSRF probes when tools are executed for real."""

    host: str = "127.0.0.1"
    port: int = 0
    path: str = "/canary/ssrf"


@dataclass
class Config:
    """A whole run's settings."""

    target: TargetConfig = field(default_factory=TargetConfig)
    limits: Limits = field(default_factory=Limits)
    report: ReportConfig = field(default_factory=ReportConfig)
    #: Probe ids or prefixes to run. Empty means all of them.
    probes: list[str] = field(default_factory=list)
    #: Per-probe limit overrides, keyed by exact probe id.
    probe_limits: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Directories of user-written `.py` probes to import before running.
    probe_dirs: list[Path] = field(default_factory=list)
    #: Probes defined entirely in the config file, no Python required.
    custom_probes: list[dict[str, Any]] = field(default_factory=list)
    #: Optional canary HTTP listener for SSRF confirmation when tools execute.
    probe_server: ProbeServerConfig = field(default_factory=ProbeServerConfig)
    #: Path the config was read from, for the report header. None if defaults.
    source: Path | None = None

    def limits_for(self, probe_id: str) -> Limits:
        """Limits for one probe: the global set plus any per-probe overrides."""
        overrides = self.probe_limits.get(probe_id)
        return self.limits.merged(overrides) if overrides else self.limits

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Read `config.yaml` if present, otherwise return built-in defaults."""
        if path is None:
            path = _discover_config()
        if path is None:
            return cls()
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")

        raw = _read_yaml(path)
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        raw = _expand_env(raw)

        config = cls(source=path)
        target_raw = raw.pop("target", None)
        if target_raw:
            _apply_section(target_raw, config.target, path, "target")
        _apply_section(raw.pop("limits", {}), config.limits, path, "limits")
        _apply_section(raw.pop("report", {}), config.report, path, "report")

        config.probes = _as_str_list(raw.pop("probes", []), path, "probes")
        config.probe_dirs = [Path(p).expanduser() for p in _as_str_list(
            raw.pop("probe_dirs", []), path, "probe_dirs"
        )]

        probe_limits = raw.pop("probe_limits", {})
        if not isinstance(probe_limits, dict):
            raise ConfigError(f"{path}: probe_limits must be a mapping of probe id to limits")
        for probe_id, overrides in probe_limits.items():
            if not isinstance(overrides, dict):
                raise ConfigError(f"{path}: probe_limits.{probe_id} must be a mapping")
            config.limits.merged(overrides)
        config.probe_limits = probe_limits

        custom = raw.pop("custom_probes", [])
        if not isinstance(custom, list):
            raise ConfigError(f"{path}: custom_probes must be a list")
        config.custom_probes = custom

        _apply_section(raw.pop("probe_server", {}), config.probe_server, path, "probe_server")

        if raw:
            raise ConfigError(
                f"{path}: unknown top-level key(s): {', '.join(sorted(raw))}. "
                "Valid keys are: target, limits, report, probes, probe_limits, "
                "probe_dirs, custom_probes, probe_server"
            )

        config.probe_dirs = [_resolve_against(p, path) for p in config.probe_dirs]
        if config.report.jsonl:
            config.report.jsonl = _resolve_against(config.report.jsonl, path)
        if config.report.docx:
            config.report.docx = _resolve_against(config.report.docx, path)
        return config


def validate_target(target: TargetConfig) -> None:
    """Require a non-empty url and model (from config and/or CLI)."""
    if not target.url.strip():
        raise ConfigError(
            "target url is required — pass --target or set target.url in config.yaml"
        )
    if not target.model.strip():
        raise ConfigError(
            "target model is required — pass --model or set target.model in config.yaml"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> Any:
    """Parse a YAML file."""
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc


def _discover_config() -> Path | None:
    """Find a `config.yaml` in the working directory, if there is one."""
    for name in DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate
    return None


def _expand_env(value: Any) -> Any:
    """Recursively replace `${VAR}` in every string with its environment value.

    A variable that is not set expands to an empty string, matching shell
    behaviour, so an absent optional API key is simply blank.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _apply_section(raw: Any, section: Any, path: Path, name: str) -> None:
    """Copy a mapping onto a dataclass, coercing types and rejecting typos."""
    if not raw:
        return
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: {name} must be a mapping")
    valid = {f.name: f for f in fields(section)}
    unknown = set(raw) - set(valid)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) in {name}: {', '.join(sorted(unknown))}. "
            f"Valid keys are: {', '.join(sorted(valid))}"
        )
    for key, value in raw.items():
        setattr(section, key, _coerce(value, valid[key].type, path, f"{name}.{key}"))


def _coerce(value: Any, annotation: Any, path: Path, where: str) -> Any:
    """Convert a YAML scalar to what the dataclass field declares.

    Annotations arrive as strings because of `from __future__ import
    annotations`, so this matches on their text rather than on the type object.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if value is None:
        return None
    try:
        if "Path" in text:
            return Path(str(value)).expanduser()
        if text.startswith("int"):
            return int(value)
        if text.startswith("float"):
            return float(value)
        if text.startswith("bool"):
            return bool(value)
        if text.startswith("str"):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {where} has the wrong type: {exc}") from exc
    return value


def _as_str_list(value: Any, path: Path, name: str) -> list[str]:
    """Accept either a single string or a list of them."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ConfigError(f"{path}: {name} must be a string or a list of strings")


def _resolve_against(target: Path, config_path: Path) -> Path:
    """Interpret a relative path as relative to the config file that named it."""
    return target if target.is_absolute() else (config_path.parent / target).resolve()
