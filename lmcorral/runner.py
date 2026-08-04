"""Runs probes against a target, monitor in hand, and writes the report."""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .protocol import Finding, Outcome, Probe, Target, Transcript

console = Console()


class Runner:
    def __init__(self, target: Target, *, verbose: bool = False) -> None:
        self.target = target
        self.verbose = verbose

    def run(self, probe: Probe) -> Finding:
        if probe.needs_tools and "tools" not in self.target.capabilities():
            model = self.target.model
            return probe.finding(Outcome.SKIP, f"target has no tool support for {model!r}")

        started = time.monotonic()
        transcripts: list[Transcript] = []
        monitors = probe.monitors()

        try:
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
        except Exception as exc:  # noqa: BLE001 - a broken probe must not kill the run
            finding = probe.finding(Outcome.ERROR, f"{type(exc).__name__}: {exc}")
            finding.transcripts = transcripts
            finding.duration_s = time.monotonic() - started
            return finding

        finding = probe.judge(transcripts)
        finding.transcripts = transcripts
        finding.duration_s = time.monotonic() - started
        return finding


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(value, "name") and not isinstance(value, (int, str)):
        return value.value  # StrEnum
    return value


def write_report(findings: list[Finding], path: Path, *, target_desc: str) -> None:
    import json

    lines = [
        json.dumps(
            {
                "type": "run",
                "at": datetime.now(UTC).isoformat(),
                "target": target_desc,
                "probe_count": len(findings),
            }
        )
    ]
    for finding in findings:
        record = _jsonable(finding)
        record["type"] = "finding"
        lines.append(json.dumps(record, default=str))
    path.write_text("\n".join(lines) + "\n")


_COLOR = {
    Outcome.PASS: "green",
    Outcome.FAIL: "red",
    Outcome.WARN: "yellow",
    Outcome.ERROR: "magenta",
    Outcome.SKIP: "dim",
}


def print_summary(findings: list[Finding]) -> None:
    table = Table(title="LMCorral results", show_lines=False)
    table.add_column("Outcome", width=8)
    table.add_column("Probe")
    table.add_column("Detail", overflow="fold")

    for finding in findings:
        color = _COLOR.get(finding.outcome, "white")
        table.add_row(
            f"[{color}]{finding.outcome.value.upper()}[/{color}]",
            finding.probe,
            finding.detail,
        )
    console.print(table)

    failed = sum(1 for f in findings if f.outcome is Outcome.FAIL)
    errored = sum(1 for f in findings if f.outcome is Outcome.ERROR)
    passed = sum(1 for f in findings if f.outcome is Outcome.PASS)
    console.print(
        f"\n{passed} passed, {failed} failed, {errored} errored, "
        f"{sum(1 for f in findings if f.outcome is Outcome.SKIP)} skipped"
    )
