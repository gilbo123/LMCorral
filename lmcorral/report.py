"""Turning findings into output: terminal table, JSONL, and Word.

Three renderers over the same list of `Finding`s. JSONL is the machine-readable
record of record — one run header line, then one line per finding, everything
the probe saw included. The terminal table is the at-a-glance summary. The Word
document is for the reader who wants a report to hand to someone who will never
open a terminal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Config
from .protocol import Finding, Outcome

console = Console()

_COLOR = {
    Outcome.PASS: "green",
    Outcome.FAIL: "red",
    Outcome.WARN: "yellow",
    Outcome.ERROR: "magenta",
    Outcome.SKIP: "dim",
}

# Word has no ANSI, so severities map to document-friendly colours instead.
_DOCX_OUTCOME_RGB = {
    Outcome.PASS: (0x2E, 0x7D, 0x32),
    Outcome.FAIL: (0xC6, 0x28, 0x28),
    Outcome.WARN: (0xEF, 0x6C, 0x00),
    Outcome.ERROR: (0x6A, 0x1B, 0x9A),
    Outcome.SKIP: (0x75, 0x75, 0x75),
}


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and enums into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _truncate_transcripts(finding: dict[str, Any], limit: int) -> None:
    """Cap the stored text on each transcript so a runaway probe stays readable.

    Mutates the already-jsonable dict in place. A runaway probe can emit tens of
    thousands of characters; keeping all of it would bloat every report for no
    added insight beyond the fact that it ran long.
    """
    for transcript in finding.get("transcripts", []):
        for key in ("text", "reasoning"):
            text = transcript.get(key) or ""
            if len(text) > limit:
                transcript[key] = text[:limit] + f"... [+{len(text) - limit} chars truncated]"


def summary_counts(findings: list[Finding]) -> dict[str, int]:
    """Tally findings by outcome, for headers and exit codes."""
    counts = {outcome.value: 0 for outcome in Outcome}
    for finding in findings:
        counts[finding.outcome.value] += 1
    return counts


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #


def print_summary(findings: list[Finding]) -> None:
    """Print the results table and a one-line tally to the terminal."""
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

    counts = summary_counts(findings)
    console.print(
        f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['warn']} warned, "
        f"{counts['error']} errored, {counts['skip']} skipped"
    )


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


def write_jsonl(findings: list[Finding], path: Path, *, config: Config, target_desc: str) -> None:
    """Write the newline-delimited JSON record: a run header, then one finding
    per line, transcripts truncated to the configured cap."""
    header = {
        "type": "run",
        "at": datetime.now(timezone.utc).isoformat(),
        "target": target_desc,
        "config_source": str(config.source) if config.source else None,
        "probe_count": len(findings),
        "summary": summary_counts(findings),
    }
    lines = [json.dumps(header)]
    for finding in findings:
        record = _jsonable(finding)
        _truncate_transcripts(record, config.report.max_transcript_chars)
        record["type"] = "finding"
        lines.append(json.dumps(record, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #


def write_docx(findings: list[Finding], path: Path, *, config: Config, target_desc: str) -> None:
    """Write a formatted Word report.

    Needs `python-docx`, which is an optional extra so that the common case — run
    probes, read the terminal, keep the JSONL — carries no heavy dependency. The
    caller is expected to have checked availability with `docx_available()` and
    produced a friendlier message; this still raises a clear error if not.
    """
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor
    except ImportError as exc:  # pragma: no cover - guarded by docx_available()
        raise RuntimeError(
            "Word output needs python-docx; install it with `pip install lmcorral[docx]` "
            "or `pip install python-docx`"
        ) from exc

    counts = summary_counts(findings)
    document = Document()

    document.add_heading("LMCorral Report", level=0)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = subtitle.add_run(datetime.now().strftime("%Y-%m-%d %H:%M"))
    run.italic = True

    meta = document.add_paragraph()
    meta.add_run("Target: ").bold = True
    meta.add_run(target_desc)
    if config.source:
        meta.add_run("\nConfig: ").bold = True
        meta.add_run(str(config.source))

    # A one-line verdict banner so the headline outcome is unmissable.
    banner = document.add_paragraph()
    banner_run = banner.add_run(
        f"{counts['fail']} failed · {counts['warn']} warned · {counts['pass']} passed · "
        f"{counts['error']} errored · {counts['skip']} skipped"
    )
    banner_run.bold = True
    banner_run.font.size = Pt(13)
    verdict_color = _DOCX_OUTCOME_RGB[Outcome.FAIL if counts["fail"] else Outcome.PASS]
    banner_run.font.color.rgb = RGBColor(*verdict_color)

    # Summary table.
    document.add_heading("Summary", level=1)
    table = document.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    header_cells = table.rows[0].cells
    for cell, label in zip(header_cells, ("Outcome", "Probe", "Severity", "Detail")):
        cell.paragraphs[0].add_run(label).bold = True
    for finding in findings:
        cells = table.add_row().cells
        outcome_run = cells[0].paragraphs[0].add_run(finding.outcome.value.upper())
        outcome_run.bold = True
        outcome_run.font.color.rgb = RGBColor(*_DOCX_OUTCOME_RGB[finding.outcome])
        cells[1].text = finding.probe
        cells[2].text = finding.severity
        cells[3].text = finding.detail

    # One detailed section per finding.
    document.add_heading("Findings", level=1)
    for finding in findings:
        document.add_heading(finding.probe, level=2)

        line = document.add_paragraph()
        status = line.add_run(finding.outcome.value.upper())
        status.bold = True
        status.font.color.rgb = RGBColor(*_DOCX_OUTCOME_RGB[finding.outcome])
        line.add_run(f"  ·  severity {finding.severity}")
        if finding.owasp:
            line.add_run(f"  ·  {finding.owasp}")

        document.add_paragraph(finding.detail)

        if finding.evidence:
            document.add_paragraph("Evidence", style="Intense Quote")
            evidence = _jsonable(finding.evidence)
            code = document.add_paragraph()
            code_run = code.add_run(json.dumps(evidence, indent=2, default=str))
            code_run.font.name = "Courier New"
            code_run.font.size = Pt(8)

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))


def docx_available() -> bool:
    """True if python-docx is importable, so the CLI can warn before running."""
    try:
        import docx  # noqa: F401
    except ImportError:
        return False
    return True
