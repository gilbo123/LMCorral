"""`lmcorral` command line.

`run` executes probes. `probes` lists what is available. Settings live in
`lmcorral.yaml` in the current directory — edit that file, then run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .config import Config, ConfigError
from .probes import all_probes, load_declarative, load_probe_dirs, select
from .report import docx_available, print_summary, summary_counts, write_docx, write_jsonl
from .runner import Runner
from .targets import build_target

console = Console()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "probes":
        return _cmd_probes(args)
    return _cmd_run(args)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all subcommands."""
    parser = argparse.ArgumentParser(prog="lmcorral", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run probes against a live endpoint")
    run_p.add_argument(
        "--config",
        type=Path,
        help="path to lmcorral.yaml (default: lmcorral.yaml in the current directory)",
    )
    run_p.add_argument(
        "--target",
        help="base URL of the endpoint (Ollama native or OpenAI-compatible)",
    )
    run_p.add_argument("--model", help="model name; default is whatever the server has loaded")
    run_p.add_argument("--api-key", help="bearer token, for OpenAI-compatible targets")
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
        "--config", type=Path, help="load lmcorral.yaml first (for custom probes in config)"
    )
    probes_p.add_argument(
        "--probe-dir", action="append", dest="probe_dirs", type=Path, help="load these dirs first"
    )

    return parser


def _load_config(args: argparse.Namespace) -> Config:
    """Read lmcorral.yaml and layer command-line overrides on top."""
    config = Config.load(getattr(args, "config", None))

    if getattr(args, "target", None):
        config.target.url = args.target
    if getattr(args, "model", None):
        config.target.model = args.model
    if getattr(args, "api_key", None):
        config.target.api_key = args.api_key
    if getattr(args, "probes", None):
        config.probes = args.probes
    if getattr(args, "probe_dirs", None):
        config.probe_dirs.extend(args.probe_dirs)
    if getattr(args, "out", None):
        config.report.jsonl = args.out
    if getattr(args, "docx", None):
        config.report.docx = args.docx
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


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the selected probes and write the configured reports."""
    try:
        config = _load_config(args)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    if config.source:
        console.print(f"[dim]config: {config.source}[/dim]")

    if config.report.docx and not docx_available():
        console.print(
            "[red]report.docx is set but python-docx is not installed.[/red] "
            "Run: pip install \"lmcorral[docx]\""
        )
        return 2

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
    runner = Runner(target, limits=config.limits, verbose=args.verbose)
    findings = []
    for probe in probes:
        console.print(f"[bold cyan]{probe.id}[/bold cyan] — {probe.summary}")
        findings.append(runner.run(probe, limits=config.limits_for(probe.id)))

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
