"""Runs probes against a target, monitor in hand.

The runner owns the turn loop and nothing else: it hands each probe its limits,
plays turns until the probe stops or `max_turns` is hit, and lets monitors abort
a stream mid-flight. Judging and reporting live elsewhere, so this stays the one
place that has to reason about the sequence of network calls.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from rich.console import Console

from .config import Limits, ProbeServerConfig
from .canary_server import CanaryServer
from .protocol import Finding, Outcome, Probe, Target, Transcript
from .report import annotate_trial_counts

console = Console()


class Runner:
    """Executes probes against one target, applying one set of limits."""

    def __init__(
        self,
        target: Target,
        *,
        limits: Limits,
        verbose: bool = False,
        probe_server: ProbeServerConfig | None = None,
        canary_server: CanaryServer | None = None,
    ) -> None:
        """`limits` is the fallback for probes without a per-probe override."""
        self.target = target
        self.limits = limits
        self.verbose = verbose
        self.probe_server = probe_server if probe_server is not None else ProbeServerConfig()
        self.canary_server = canary_server

    def run(self, probe: Probe, *, limits: Limits | None = None) -> Finding:
        """Run one probe end to end and return its verdict.

        A broken probe is caught and turned into an ERROR finding rather than
        being allowed to abort the whole run — one probe's bug should not cost
        the results of every probe after it.
        """
        probe.configure(limits or self.limits, self.probe_server, self.canary_server)

        if probe.needs_tools and "tools" not in self.target.capabilities():
            model = self.target.model
            return probe.finding(Outcome.SKIP, f"target has no tool support for {model!r}")

        started = time.monotonic()
        transcripts: list[Transcript] = []
        monitors = probe.monitors()

        try:
            transcripts = self._play(probe, monitors)
        except Exception as exc:  # noqa: BLE001 - a broken probe must not kill the run
            finding = probe.finding(Outcome.ERROR, f"{type(exc).__name__}: {exc}")
            finding.transcripts = transcripts
            finding.duration_s = time.monotonic() - started
            annotate_trial_counts(finding)
            return finding

        finding = probe.judge(transcripts)
        finding.transcripts = transcripts
        finding.duration_s = time.monotonic() - started
        annotate_trial_counts(finding)
        return finding

    def _play(self, probe: Probe, monitors: Sequence) -> list[Transcript]:
        """Drive the probe's turns, then its `follow_up` chain, up to max_turns.

        Scripted turns from `turns()` are played first; once they run out, the
        probe's `follow_up` decides each subsequent turn from what it has seen.
        `max_turns` caps the total so a retry-storm probe cannot itself become
        the runaway loop it is testing for.
        """
        transcripts: list[Transcript] = []
        turns = iter(probe.turns())
        turn = next(turns, None)
        while turn is not None and len(transcripts) < probe.max_turns:
            if self.verbose:
                console.print(f"  [dim]-> {probe.id} / {turn.label}[/dim]")
            transcript = self.target.stream(turn, monitors)
            transcripts.append(transcript)
            probe.after_turn(self.target, transcript)
            if self.verbose and transcript.signals:
                for signal in transcript.signals:
                    console.print(f"     [yellow]{signal}[/yellow]")

            next_turn = next(turns, None)
            if next_turn is None:
                next_turn = probe.follow_up(transcripts)
            turn = next_turn
        return transcripts
