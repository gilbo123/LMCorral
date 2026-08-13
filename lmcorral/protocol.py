"""The test protocol.

Four things, and adding a new test means implementing only the last one:

    Target     something that streams tokens and can be hung up on mid-generation
    Monitor    watches a stream as it arrives and may demand it be aborted
    Probe      declares what to send, which monitors to arm, and how to judge
    Finding    the verdict, one per probe

The load-bearing idea is that a Monitor runs *during* generation rather than
after it. Offline scanners can only tell you a model produced 40,000 tokens of
garbage; a Monitor can end it at token 400. That is the difference between a
test harness and a circuit breaker, and it is why the same objects serve both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Limits, ProbeServerConfig
from .canary_server import CanaryServer

# --------------------------------------------------------------------------- #
# What a probe sends
# --------------------------------------------------------------------------- #


@dataclass
class Turn:
    """A single request to put to the target."""

    messages: list[dict[str, Any]]
    label: str = ""
    tools: list[dict[str, Any]] | None = None
    #: Passed through to the target verbatim (`num_predict`, `temperature`, ...).
    options: dict[str, Any] = field(default_factory=dict)
    #: Probes that deliberately want no output ceiling say so here, so that a
    #: global default cap does not silently make a runaway test unfalsifiable.
    uncapped: bool = False


# --------------------------------------------------------------------------- #
# What monitors say
# --------------------------------------------------------------------------- #


class _Str(str, Enum):
    """A str-valued enum.

    `enum.StrEnum` would be the obvious choice but it landed in 3.11, and the
    machines this tool is aimed at are frequently older than that — RHEL 9 still
    ships 3.9. The mixin gives the same behaviour everywhere.
    """

    def __str__(self) -> str:
        return self.value


class Action(_Str):
    """What a monitor wants done with the stream it is watching."""

    CONTINUE = "continue"
    FLAG = "flag"
    """Record it and keep generating."""
    ABORT = "abort"
    """Hang up now. The corral gate closes."""


@dataclass
class Signal:
    """A monitor's ruling on a stream in progress."""

    action: Action
    monitor: str
    reason: str
    at_token: int = 0
    at_second: float = 0.0

    def __str__(self) -> str:
        """One-line form used in the terminal and in report evidence."""
        return f"[{self.monitor}] {self.reason} (token {self.at_token}, {self.at_second:.1f}s)"


@dataclass
class StreamView:
    """A monitor's read-only window onto a generation in progress."""

    delta: str
    """Text that arrived in this chunk (visible output or reasoning)."""
    text: str
    """User-visible output accumulated so far."""
    index: int
    """Number of chunks received, a stand-in for token count."""
    elapsed_s: float
    reasoning: str = ""
    """Thinking trace accumulated so far, when the endpoint exposes one."""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    @property
    def all_text(self) -> str:
        """Visible output plus any reasoning trace."""
        return self.text + self.reasoning


class Monitor(ABC):
    """Watches one generation. Instances are not reused across turns."""

    name: str = "monitor"

    def reset(self) -> None:  # noqa: B027 - optional hook, empty by design
        """Called before each turn."""

    @abstractmethod
    def observe(self, view: StreamView) -> Signal | None:
        """Return a Signal to flag or abort, or None to allow the stream on."""


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


@dataclass
class Transcript:
    """Everything observed during one turn."""

    label: str
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    chunks: int = 0
    elapsed_s: float = 0.0
    ttft_s: float | None = None
    aborted: bool = False
    signals: list[Signal] = field(default_factory=list)
    error: str | None = None
    #: Whatever the server volunteered on the final chunk (eval counts, etc).
    server: dict[str, Any] = field(default_factory=dict)

    @property
    def abort_signal(self) -> Signal | None:
        """The signal that ended the stream, if a monitor aborted it."""
        return next((s for s in self.signals if s.action is Action.ABORT), None)

    @property
    def tokens_per_s(self) -> float:
        """Throughput in chunks per second, or 0 if nothing was generated."""
        return self.chunks / self.elapsed_s if self.elapsed_s > 0 else 0.0


class Outcome(_Str):
    """The verdict of a probe."""

    PASS = "pass"
    """The model, or our gate, behaved as it must."""
    FAIL = "fail"
    """A limit was crossed that nothing stopped."""
    WARN = "warn"
    """Survivable, but somebody should look."""
    ERROR = "error"
    SKIP = "skip"


@dataclass
class Finding:
    """One probe's verdict, plus the evidence and transcripts behind it."""

    probe: str
    outcome: Outcome
    detail: str
    severity: str = "medium"
    owasp: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    transcripts: list[Transcript] = field(default_factory=list)
    duration_s: float = 0.0
    trials_passed: int = 0
    trials_total: int = 0
    turn_closure: dict[str, str] | None = None


# --------------------------------------------------------------------------- #
# What a probe is
# --------------------------------------------------------------------------- #


class Probe(ABC):
    """One test.

    Subclass, set the class attributes, implement `turns` and `judge`, and
    decorate with `@register`. Nothing else in the codebase needs to change.

    Tunable numbers come from `self.limits`, which the runner replaces with the
    user's configuration before the probe runs. Reading them inside `turns` and
    `monitors` rather than in `__init__` is what makes that possible.
    """

    id: str = ""
    summary: str = ""
    owasp: str = ""
    severity: str = "medium"
    tags: tuple[str, ...] = ()
    #: Probes that only make sense against a tool-calling model set this so the
    #: runner can skip rather than report a misleading pass.
    needs_tools: bool = False
    #: Ceiling on `follow_up` rounds. A probe that tests for runaway loops must
    #: not be able to become one.
    max_turns: int = 12

    #: Replaced by ``configure()`` before a probe runs.
    limits: Limits | None = None
    probe_server: ProbeServerConfig = ProbeServerConfig()
    canary_server: CanaryServer | None = None

    def configure(
        self,
        limits: Limits,
        probe_server: ProbeServerConfig | None = None,
        canary_server: CanaryServer | None = None,
    ) -> None:
        """Adopt the caller's limits and optional SSRF canary HTTP listener."""
        self.limits = limits
        self.probe_server = probe_server if probe_server is not None else ProbeServerConfig()
        self.canary_server = canary_server

    @abstractmethod
    def turns(self) -> Iterable[Turn]:
        """The requests this probe sends, in order."""

    def monitors(self) -> list[Monitor]:
        """Monitors armed for every turn of this probe."""
        return []

    def follow_up(self, transcripts: Sequence[Transcript]) -> Turn | None:
        """Next turn given what has happened, or None to stop.

        This is what makes multi-turn probes possible — retry storms, gradual
        escalation, anything where the next prompt depends on the last answer.
        Single-turn probes ignore it.
        """
        return None

    def after_turn(self, target: Target, transcript: Transcript) -> None:  # noqa: B027
        """Optional: inspect the target itself once a turn is over.

        Containment probes need this. Whether a disconnect actually stopped the
        work is a fact about the server, not about the tokens we received.
        """

    @abstractmethod
    def judge(self, transcripts: Sequence[Transcript]) -> Finding:
        """Turn what happened into a single verdict."""

    def finding(
        self,
        outcome: Outcome,
        detail: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> Finding:
        """Build a Finding, filling in this probe's identity and metadata."""
        return Finding(
            probe=self.id,
            outcome=outcome,
            detail=detail,
            severity=self.severity,
            owasp=self.owasp,
            evidence=evidence or {},
        )


# --------------------------------------------------------------------------- #
# What a target is
# --------------------------------------------------------------------------- #


class Target(ABC):
    """A model endpoint that can be streamed from and hung up on."""

    name: str = "target"
    model: str = ""

    @abstractmethod
    def stream(self, turn: Turn, monitors: Sequence[Monitor]) -> Transcript:
        """Send `turn`, feed each chunk to every monitor, and abort the stream
        the moment any of them returns `Action.ABORT`."""

    def capabilities(self) -> set[str]:
        """Best-effort feature discovery, e.g. `{"tools", "thinking"}`."""
        return set()

    def health(self) -> tuple[bool, str]:
        return True, ""

    def settling_delay(self, *, timeout: float = 30.0) -> tuple[float, str]:
        """Seconds until the endpoint will serve a trivial request again.

        Measured after an abort, this is the only way to tell a stream that was
        truncated from work that was actually cancelled. Returns (delay, error).
        """
        return 0.0, "unsupported"
