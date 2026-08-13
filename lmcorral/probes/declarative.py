"""Probes defined in YAML, no Python required.

This covers the most common shape of custom test: send one or more prompts, then
pass or fail on whether the reply contains (or is missing) some text. Anything
more subtle — inspecting tool calls, timing the server, multi-turn escalation —
is a real `Probe` subclass in a `.py` file instead, and the config points at it
with `probe_dirs`.

A declarative probe looks like this in `config.yaml`:

    custom_probes:
      - id: custom.no_markdown
        summary: Model wraps code in markdown fences when asked not to
        severity: low
        prompts:
          - "Return only raw JSON, with no markdown formatting: {\"ok\": true}"
        expect:
          # The check that decides PASS vs FAIL. Exactly one of these keys.
          must_not_contain: ["```"]
        match: all        # 'all' (default): every prompt must satisfy the check.
                          # 'any': at least one prompt must satisfy it.
        ignore_case: true

Supported `expect` checks (choose one):

    must_contain:      list of strings — fail if none present
    must_not_contain:  list of strings — fail if any present
    regex:             pattern — fail if it does not match
    not_regex:         pattern — fail if it does match
    refused:           true/false — fail if the answer's refusal state differs

`refused` reuses the same refusal heuristic as the safety probes, so a
declarative probe can assert "this must be refused" or "this must be answered"
without pattern-matching wording by hand.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from ..monitors import TokenBudget, WallClock
from ..protocol import Finding, Monitor, Outcome, Probe, Transcript, Turn
from .safety import looks_like_refusal

_VALID_CHECKS = ("must_contain", "must_not_contain", "regex", "not_regex", "refused")


def build_declarative_probe(spec: dict[str, Any]) -> type[Probe]:
    """Compile one `custom_probes` mapping into a Probe subclass.

    Validation is strict and eager: an unknown key or a missing check raises
    here, at config-load time, rather than producing a probe that quietly tests
    nothing.
    """
    if not isinstance(spec, dict):
        raise ValueError("each custom probe must be a mapping")

    probe_id = spec.get("id")
    if not probe_id or not isinstance(probe_id, str):
        raise ValueError("custom probe needs a string 'id'")

    prompts = spec.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"{probe_id}: 'prompts' must be a non-empty list of strings")

    expect = spec.get("expect")
    if not isinstance(expect, dict) or len(expect) != 1:
        raise ValueError(
            f"{probe_id}: 'expect' must be a mapping with exactly one check "
            f"(one of: {', '.join(_VALID_CHECKS)})"
        )
    (check_name, check_value), = expect.items()
    if check_name not in _VALID_CHECKS:
        raise ValueError(
            f"{probe_id}: unknown check {check_name!r}; valid checks are {', '.join(_VALID_CHECKS)}"
        )

    validated_owasp = spec.get("owasp")
    if not validated_owasp or not isinstance(validated_owasp, str):
        raise ValueError(
            f"{probe_id}: 'owasp' is required (e.g. LLM01:2025 Prompt Injection)"
        )

    match_mode = spec.get("match", "all")
    if match_mode not in ("any", "all"):
        raise ValueError(f"{probe_id}: 'match' must be 'any' or 'all'")

    ignore_case = bool(spec.get("ignore_case", True))
    system = spec.get("system")
    if system is not None and not isinstance(system, str):
        raise ValueError(f"{probe_id}: 'system' must be a string")

    unknown = set(spec) - {
        "id", "summary", "severity", "owasp", "tags", "prompts", "expect",
        "match", "ignore_case", "system", "temperature",
    }
    if unknown:
        raise ValueError(f"{probe_id}: unknown key(s): {', '.join(sorted(unknown))}")

    # Precompile the checker once, closed over the validated spec.
    checker = _make_checker(check_name, check_value, ignore_case)

    class DeclarativeProbe(Probe):
        id = probe_id
        summary = spec.get("summary", f"custom probe {probe_id}")
        severity = spec.get("severity", "medium")
        owasp = validated_owasp
        tags = tuple(spec.get("tags", ("custom",)))

        def turns(self) -> Iterable[Turn]:
            """One request per configured prompt, optionally with a system message."""
            temperature = spec.get("temperature", self.limits.temperature)
            for i, prompt in enumerate(prompts):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                yield Turn(
                    messages=messages,
                    label=f"prompt-{i}",
                    options={"num_predict": self.limits.max_tokens, "temperature": temperature},
                )

        def monitors(self) -> list[Monitor]:
            """Standard ceilings; declarative probes judge on content, not timing."""
            return [
                TokenBudget(self.limits.token_budget),
                WallClock(self.limits.wall_clock_seconds),
            ]

        def judge(self, transcripts: Sequence[Transcript]) -> Finding:
            """Apply the single configured check under the chosen match mode."""
            results = {t.label: checker(t.text) for t in transcripts}
            passed = [label for label, ok in results.items() if ok]
            failed = [label for label, ok in results.items() if not ok]

            # 'all': the probe passes only if every prompt satisfied the check.
            # 'any': one satisfied prompt is enough.
            overall_ok = not failed if match_mode == "all" else bool(passed)

            evidence = {
                "check": check_name,
                "match": match_mode,
                "results": results,
                "openings": {t.label: t.text[:200].replace("\n", " ") for t in transcripts},
            }

            if overall_ok:
                return self.finding(
                    Outcome.PASS,
                    f"{check_name} check satisfied ({match_mode} of {len(transcripts)} prompt(s))",
                    evidence=evidence,
                )
            return self.finding(
                Outcome.FAIL,
                f"{check_name} check failed on {len(failed)}/{len(transcripts)} prompt(s): "
                f"{', '.join(failed)}",
                evidence=evidence,
            )

    DeclarativeProbe.__name__ = f"Declarative_{re.sub(r'[^0-9A-Za-z]+', '_', probe_id)}"
    return DeclarativeProbe


def _make_checker(name: str, value: Any, ignore_case: bool):
    """Return a `text -> bool` that is True when the answer *passes* the check."""
    if name in ("must_contain", "must_not_contain"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{name} must be a list of strings")
        needles = [v.lower() for v in value] if ignore_case else list(value)

        def check(text: str) -> bool:
            hay = text.lower() if ignore_case else text
            present = any(n in hay for n in needles)
            return present if name == "must_contain" else not present

        return check

    if name in ("regex", "not_regex"):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string pattern")
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(value, flags)

        def check(text: str) -> bool:
            found = pattern.search(text) is not None
            return found if name == "regex" else not found

        return check

    # name == "refused"
    want_refusal = bool(value)

    def check(text: str) -> bool:
        return looks_like_refusal(text) == want_refusal

    return check
