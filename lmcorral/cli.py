"""`lmcorral` command line.

`run` executes probes. `probes` lists what is available. `report` converts a
JSONL file to Word.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .canary_server import CanaryServer
from .config import Config, ConfigError, validate_target
from .probes import all_probes, load_declarative, load_probe_dirs, select
from .report import (
    print_summary,
    summary_counts,
    write_docx,
    write_docx_from_jsonl,
    write_jsonl,
)
from .runner import Runner
from .targets import build_target

console = Console()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "probes":
        return _cmd_probes(args)
    if args.command == "report":
        return _cmd_report(args)
    return _cmd_run(args)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all subcommands."""
    parser = argparse.ArgumentParser(prog="lmcorral", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run probes against a live endpoint")
    run_p.add_argument(
        "--config",
        type=Path,
        help="path to config.yaml (default: config.yaml in the current directory, if present)",
    )
    run_p.add_argument(
        "--target",
        help="model server base URL (overrides config target.url)",
    )
    run_p.add_argument(
        "--model",
        help="model name (overrides config target.model)",
    )
    run_p.add_argument(
        "--probe",
        action="append",
        dest="probes",
        help="probe id or prefix to run; repeatable; overrides config; default is all",
    )
    run_p.add_argument(
        "--probe-dir",
        action="append",
        dest="probe_dirs",
        type=Path,
        help="directory of user-written .py probe files to load; repeatable",
    )
    run_p.add_argument("--out", type=Path, help="JSONL report path (overrides config)")
    run_p.add_argument("--docx", type=Path, help="also write a Word report to this path")
    run_p.add_argument("--verbose", action="store_true", help="print each turn and signal")

    probes_p = sub.add_parser("probes", help="list available probes")
    probes_p.add_argument(
        "--config",
        type=Path,
        help="path to config.yaml (optional; loads custom probes from config)",
    )
    probes_p.add_argument(
        "--probe-dir", action="append", dest="probe_dirs", type=Path, help="load these dirs first"
    )

    report_p = sub.add_parser("report", help="convert a JSONL report to Word")
    report_p.add_argument(
        "jsonl",
        type=Path,
        help="path to lmcorral-report.jsonl (or any JSONL written by lmcorral run)",
    )
    report_p.add_argument(
        "--docx",
        type=Path,
        help="Word output path (default: same name as the JSONL with .docx extension)",
    )

    return parser


def _load_config(args: argparse.Namespace, *, require_target: bool = False) -> Config:
    """Read config and apply CLI overrides."""
    config = Config.load(getattr(args, "config", None))

    if getattr(args, "target", None):
        config.target.url = args.target
    if getattr(args, "model", None):
        config.target.model = args.model
    if getattr(args, "probes", None):
        config.probes = args.probes
    if getattr(args, "probe_dirs", None):
        config.probe_dirs.extend(args.probe_dirs)
    if getattr(args, "out", None):
        config.report.jsonl = args.out
    if getattr(args, "docx", None):
        config.report.docx = args.docx

    if require_target:
        validate_target(config.target)
    return config


def _load_custom_probes(config: Config) -> None:
    """Import user probe directories and register declarative probes from config."""
    for note in load_probe_dirs(config.probe_dirs):
        style = "yellow" if note.startswith("error") or "not found" in note else "dim"
        console.print(f"[{style}]{note}[/{style}]")
    for note in load_declarative(config.custom_probes):
        style = "yellow" if note.startswith("error") else "dim"
        console.print(f"[{style}]{note}[/{style}]")


def _cmd_probes(args: argparse.Namespace) -> int:
    """List every registered probe."""
    try:
        config = _load_config(args)
        _load_custom_probes(config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    for probe_id, cls in all_probes().items():
        tag = " [tools]" if cls.needs_tools else ""
        console.print(f"[bold]{probe_id}[/bold]{tag}  {cls.summary}")
        if cls.owasp:
            console.print(f"  [dim]{cls.owasp}[/dim]")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Convert an existing JSONL report into a Word document."""
    jsonl_path = args.jsonl
    docx_path = args.docx or jsonl_path.with_suffix(".docx")

    try:
        write_docx_from_jsonl(jsonl_path, docx_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(f"Word report: [bold]{docx_path}[/bold]")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the selected probes and write the configured reports."""
    try:
        config = _load_config(args, require_target=True)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    if config.source:
        console.print(f"[dim]config: {config.source}[/dim]")

    try:
        _load_custom_probes(config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    target = build_target(
        config.target.url,
        config.target.model,
        api_key=config.target.api_key,
        read_timeout=config.target.read_timeout_seconds,
    )
    ok, message = target.health()
    if not ok:
        console.print(f"[red]target unreachable:[/red] {message}")
        return 2
    console.print(f"[green]target ok:[/green] {message}")

    try:
        probes = select(config.probes)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(f"running {len(probes)} probe(s)...\n")
    canary = CanaryServer()
    if config.probe_server.port:
        url = canary.start(
            config.probe_server.host,
            config.probe_server.port,
            config.probe_server.path,
        )
        console.print(f"[dim]canary server: {url}[/dim]")
    runner = Runner(
        target,
        limits=config.limits,
        verbose=args.verbose,
        probe_server=config.probe_server,
        canary_server=canary if config.probe_server.port else None,
    )
    findings = []
    try:
        for probe in probes:
            console.print(f"[bold cyan]{probe.id}[/bold cyan] — {probe.summary}")
            findings.append(runner.run(probe, limits=config.limits_for(probe.id)))
    finally:
        canary.stop()

    print_summary(findings)

    if config.report.jsonl:
        write_jsonl(findings, config.report.jsonl, config=config, target_desc=message)
        console.print(f"\nJSONL report: [bold]{config.report.jsonl}[/bold]")
    if config.report.docx:
        write_docx(findings, config.report.docx, config=config, target_desc=message)
        console.print(f"Word report: [bold]{config.report.docx}[/bold]")

    counts = summary_counts(findings)
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
