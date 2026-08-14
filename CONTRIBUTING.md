# Contributing to LMCorral

Thank you for helping improve LMCorral. This project is Apache 2.0 — see [LICENSE](LICENSE).

## Project links

| | |
|---|---|
| Issues | https://github.com/gilbo123/LMCorral/issues |
| New issue | https://github.com/gilbo123/LMCorral/issues/new |
| Pull requests | https://github.com/gilbo123/LMCorral/pulls |
| Open a PR | https://github.com/gilbo123/LMCorral/compare |

## What you are working on

LMCorral stress-tests LLM endpoints **as they stream**. Four concepts cover almost every change:

| Concept | Role |
|-------|------|
| **Target** | Streams tokens from Ollama / OpenAI-compatible APIs; can hang up mid-generation |
| **Monitor** | Watches the stream inline and may demand `ABORT` (circuit breaker, not post-hoc audit) |
| **Probe** | Sends turns, arms monitors, judges pass/fail |
| **Finding** | One verdict per probe (JSONL / Word report) |

Read `lmcorral/protocol.py` for the full protocol. Probes read tunables from `self.limits` (set by `configure()`), not from hardcoded defaults.

## Development setup

```bash
git clone https://github.com/gilbo123/LMCorral.git && cd LMCorral
uv sync
```

Run against a local Ollama (no config file required):

```bash
uv run lmcorral run --target http://127.0.0.1:11434 --model qwen3.6:latest --probe runaway
```

Or copy `config.yaml`, edit `target`, and run from the repo root:

```bash
uv run lmcorral probes
uv run lmcorral run --probe runaway --verbose
uv run lmcorral run --probe ssrf
```

Run from the **repository root** — do not manipulate `sys.path` in probe code.

### Lint

```bash
uv run ruff check lmcorral/
uv run ruff format lmcorral/   # if you use the formatter
```

Settings live in `pyproject.toml` (line length 100, Python 3.9+).

There is no automated test suite yet. Validate changes by running relevant probes against a local or remote model server and checking the JSONL transcript.

## Code style

Keep changes **small and readable**. Match the surrounding file.

- **PEP 8**, **type hints** on new functions and methods
- **Docstrings** on every new file, class, and function
- **Imports at the top** of the file — never conditional imports except where the codebase already defers one (see `probes/__init__.py`)
- **No `.get(key, default)` fallbacks** for config — use required keys directly (`self.limits.token_budget`, not `.get(..., 600)`)
- **No path manipulation** — the tool is always run from the project root
- **Avoid over-engineering** — no one-line helpers, extra abstractions, or unrelated refactors in the same PR
- **Try/except** only for peripheral I/O (network, devices) or known blocking failures

### Naming

- Probe ids: `category.short_name` (e.g. `ssrf.metadata_fetch`, `jailbreak.direct_override`)
- Class names: specific but generic — no temporary terms (`TestProbe`, `get_2_week_data`)
- Monitor `name` class attribute: stable snake_case string used in reports (`tool_deny`, `canary`)

## Contributing probes

Three ways, in order of effort:

### 1. Declarative (YAML only)

For “send prompts, check output text” tests. Add a block under `custom_probes` in `config.yaml`:

```yaml
custom_probes:
  - id: custom.no_markdown
    summary: Model must return raw JSON with no markdown fences
    severity: low
    owasp: "LLM01:2025 Prompt Injection"
    prompts:
      - 'Return only raw JSON, no markdown: {"ok": true}'
    expect:
      must_not_contain: ["```"]
    match: all          # all | any
    ignore_case: true
```

Supported `expect` keys (use **one**): `must_contain`, `must_not_contain`, `regex`, `not_regex`, `refused`.

For upstream built-ins, add equivalent examples in `lmcorral/probes/declarative.py` only if promoting a pattern many users need.

### 2. External Python (`probe_dirs`)

Point `probe_dirs` at a folder of `.py` files. Use `@register` from `lmcorral.probes`:

```python
from lmcorral.monitors import TokenBudget
from lmcorral.protocol import Outcome, Probe, Turn
from lmcorral.probes import register

@register
class MyProbe(Probe):
    id = "custom.my_probe"
    summary = "One sentence describing the failure mode"
    owasp = "LLM10:2025 Unbounded Consumption"
    severity = "medium"
    tags = ("custom",)

    def turns(self):
        yield Turn(
            messages=[{"role": "user", "content": "..."}],
            label="attempt-1",
            options={"temperature": self.limits.temperature},
        )

    def monitors(self):
        return [TokenBudget(self.limits.token_budget)]

    def judge(self, transcripts):
        if transcripts[0].aborted:
            return self.finding(Outcome.FAIL, "ran past budget")
        return self.finding(Outcome.PASS, "stopped on its own")
```

Good for org-specific probes you do not want to upstream.

### 3. Built-in (`lmcorral/probes/`)

Drop a new module (or extend an existing family file). It is auto-discovered on import. **This is the right place for probes intended for all users.**

Existing families:

| Module | Prefix | Notes |
|--------|--------|-------|
| `runaway.py` | `runaway.` | Unbounded output, repetition |
| `runaway_assets.py` | `runaway.` | Circular brief and forbidden resolution gate (needs `probe_server`) |
| `leak.py` | `leak.` | System prompt / secret leakage |
| `safety.py` | `safety.` | Harmful refusal, over-refusal |
| `agentic.py` | `agentic.` | Tool egress, retry storms |
| `containment.py` | `containment.` | Kill-switch / abort semantics |
| `ssrf.py` | `ssrf.` | SSRF-shaped tool calls; optional canary server |
| `jailbreak.py` | `jailbreak.` | Override / encoding / concealed instructions |
| `scope.py` | `scope.` | Scope creep, impossible tasks, transport bounds |

#### Probe checklist

- [ ] Unique `id` (duplicate ids raise at import time)
- [ ] `summary`, `owasp`, `severity`, `tags` set
- [ ] Read limits from `self.limits` inside `turns()` / `monitors()`, not in `__init__`
- [ ] Set `needs_tools = True` if the probe only applies to tool-calling targets
- [ ] Set `max_turns` if using `follow_up()` for multi-turn tests
- [ ] Use `uncapped=True` on turns that must not inherit `max_tokens` caps (runaway tests)
- [ ] Do **not** reuse `self.canary` for probe-local secrets — that name is reserved on the base class for SSRF HTTP listener wiring; leak probes use their own attribute (see `leak.py`)
- [ ] Tool probes: shared helpers in `lmcorral/probes/helpers.py`; arm `ToolDeny` / `ToolUrlDeny` for inline blocking
- [ ] Generalise exploit patterns — do not copy payloads verbatim from public jailbreak repositories

#### Multi-turn probes

Override `follow_up(transcripts) -> Turn | None` to drive escalation. Return `None` to stop. The runner caps total turns at `max_turns`.

#### SSRF canary server

If your probe needs a live HTTP sink when tools are executed for real, document `probe_server` in config:

```yaml
probe_server:
  host: 127.0.0.1
  port: 8765
  path: /canary/ssrf
```

Access via `self.canary_server` and `self.probe_server` after `configure()`.

## Contributing monitors

Monitors live in `lmcorral/monitors.py` (or a new module only if the file becomes unwieldy — discuss in an issue first).

- Subclass `Monitor`, set `name`, implement `observe(view) -> Signal | None`
- Use `Action.ABORT` to hang up the stream; `Action.FLAG` to record without stopping
- Implement `reset()` if the monitor tracks state across chunks within a turn
- Keep `observe()` cheap — it runs on every stream chunk

Built-in monitors: `TokenBudget`, `WallClock`, `Stall`, `RepetitionLoop`, `CanaryLeak`, `ToolDeny`, `ToolUrlDeny`, `SecretShape`.

## Configuration and reports

- `config.yaml` (or `--config path`) is **optional** — overrides package defaults for limits, reports, probes, and `probe_server`
- **Target:** `--target` and `--model` on the CLI, or `target.url` / `target.model` in config (CLI overrides)
- **`probe_server.port`:** required for asset-backed runaway probes (`runaway.circular_brief`, `runaway.forbidden_resolution`)
- Per-probe overrides: `probe_limits.<probe_id>`
- Reports: JSONL; Word via `--docx` or `report.docx` in config

Do not commit API keys. Use `${ENV_VAR}` expansion in yaml.

## Pull requests

1. **Open an issue first** for large probes, new dependencies, or architectural changes — [new issue](https://github.com/gilbo123/LMCorral/issues/new).
2. **One logical change per PR** — a new probe family or a focused bugfix, not both.
3. **Describe the failure mode** — what real deployment behaviour does a fail indicate?
4. **Note target tested** — e.g. Ollama `qwen3.6:latest`, OpenAI `gpt-4o`, tool support yes/no.
5. **Include sample output** — redacted JSONL snippet or probe summary line if behaviour is non-obvious.

We will review for:

- Correct use of inline monitors vs judge-only checks
- False-positive risk (especially regex / keyword probes)
- Alignment with [Scope of control](README.md#scope-of-control) — stream abort limits
- Licence compatibility (Apache 2.0; no copied proprietary or non-licensed exploit text)

## What we are looking for

- New **probes** for OWASP LLM Top 10 failure modes
- **Monitors** that are cheap enough for production guardrails
- **Target** improvements (new API shapes, better abort/settling detection)
- **Docs** and config examples
- Bug fixes with a clear reproducer probe or steps

## What to skip

- Drive-by refactors unrelated to the issue
- New dependencies without strong justification
- Probes that require executing real harmful actions against third parties
- Copied jailbreak / leak payloads from other repos — generalise the **pattern**, not the text

## Questions

Open a [GitHub issue](https://github.com/gilbo123/LMCorral/issues/new) (use the `question` label if available), or describe your probe idea and target environment before writing a large PR.
