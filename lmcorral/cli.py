"""`lmcorral` — attach to a running model endpoint and run breakage probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .probes import all_probes, select
from .runner import Runner, print_summary, write_report
from .targets import build_target

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lmcorral")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run probes against a live endpoint")
    run_p.add_argument(
        "--target",
        default="http://127.0.0.1:11434",
        help="base URL of the endpoint (Ollama native API or OpenAI-compatible), "
        "default http://127.0.0.1:11434",
    )
    run_p.add_argument("--model", default="", help="model name; default is whatever is loaded")
    run_p.add_argument("--api-key", default="", help="bearer token, for OpenAI-compatible targets")
    run_p.add_argument(
        "--probe",
        action="append",
        dest="probes",
        help="probe id or prefix to run, e.g. --probe runaway; repeatable; default: all",
    )
    run_p.add_argument("--out", type=Path, default=Path("lmcorral-report.jsonl"))
    run_p.add_argument("--verbose", action="store_true")

    sub.add_parser("probes", help="list available probes")

    args = parser.parse_args(argv)

    if args.command == "probes":
        for probe_id, cls in all_probes().items():
            tag = " [tools]" if cls.needs_tools else ""
            console.print(f"[bold]{probe_id}[/bold]{tag}  {cls.summary}")
            console.print(f"  [dim]{cls.owasp}[/dim]")
        return 0

    target = build_target(args.target, args.model, api_key=args.api_key)
    ok, message = target.health()
    if not ok:
        console.print(f"[red]target unreachable:[/red] {message}")
        return 2
    console.print(f"[green]target ok:[/green] {message}")

    try:
        probes = select(args.probes)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(f"running {len(probes)} probe(s)...\n")
    runner = Runner(target, verbose=args.verbose)
    findings = []
    for probe in probes:
        console.print(f"[bold cyan]{probe.id}[/bold cyan] — {probe.summary}")
        finding = runner.run(probe)
        findings.append(finding)

    print_summary(findings)
    write_report(findings, args.out, target_desc=message)
    console.print(f"\nfull report: [bold]{args.out}[/bold]")

    return 1 if any(f.outcome.value == "fail" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
