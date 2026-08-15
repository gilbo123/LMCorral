<!-- Image 1: logo -->
![LMCorral logo](docs/LMCorral-logo.png)

# LMCorral

**Corral your language models.** Stress-test them outside production, and cut the stream when they bolt.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Issues](https://img.shields.io/github/issues/gilbo123/LMCorral)](https://github.com/gilbo123/LMCorral/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/gilbo123/LMCorral)](https://github.com/gilbo123/LMCorral/pulls)

### 1. Install LMCorral on your laptop or server

From source:

```bash
git clone https://github.com/gilbo123/LMCorral.git && cd LMCorral
uv sync                    # creates .venv and installs deps (recommended)
```

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**Release wheel (pip, no clone):** install the `.whl` from
[GitHub Releases](https://github.com/gilbo123/LMCorral/releases), then run:

```bash
pip install https://github.com/gilbo123/LMCorral/releases/download/v0.1.1/lmcorral-0.1.1-py3-none-any.whl
lmcorral run --target http://127.0.0.1:11434 --model qwen3.5:9b
```

No config file required — limits and reports use package defaults. Add an optional
`config.yaml` when you want to tune budgets, probe filters, or report paths.


### 2. Point LMCorral at your model server

Pass the endpoint on the command line (works from any directory after `pip install`):

```bash
lmcorral run --target http://127.0.0.1:11434 --model qwen3.5:9b
```

Or put defaults in `config.yaml` in the directory where you run the tool:

```yaml
target:
  url: http://127.0.0.1:11434
  model: "qwen3.5:9b"
```

CLI `--target` and `--model` override the config file when both are set.

### 3. Run

```bash
# pip install example
lmcorral run --target http://127.0.0.1:11434 --model qwen3.5:9b

# Using UV from a clone (optional config.yaml for overrides)
uv run lmcorral run --verbose --probe ssrf --docx --html

# Using LMCorral directly
lmcorral run --verbose --docx --html
lmcorral report lmcorral-report.jsonl
lmcorral report lmcorral-report.jsonl --docx --html

# Using LMCorral as a module
python -m lmcorral run         # module form; works with venv active (pip or uv)
```

### Table output example (`qwen3.5:9b` on Ollama)

<!-- Image 3: table report -->
![LMCorral table report](docs/LMCorral_Table.png)

The table title includes the model under test. Each row shows **Outcome**, **Trials
passed** as `passed/total` (sub-prompts or turns within that probe), and **Detail**. A
**Score** line below totals trial pass rate across probes; skipped probes are excluded.


## What it is

**LMCorral** sits in front of a model server you already run (Ollama,
vLLM, OpenAI-compatible APIs) and watches the response **as it streams**. Monitors can hang up
mid-generation — runaway output, repetition, leaked secrets, denied tool calls — without copying
the model or rebuilding your stack.

That is restraint at the wire, not isolation of the weights. Malicious artifacts at load time are
a different problem; this tool targets failures that show up once the endpoint is already
answering.

## Run checks

Built-in probes cover runaway generation, prompt leak, tool egress/retry, stream abort,
refusal in both directions, SSRF-shaped tool calls, jailbreak framing, and scope creep.
List them:

```bash
lmcorral probes
lmcorral run                 # all probes; exit 1 if any fail
lmcorral run --probe ssrf       # SSRF / chained egress (needs tool-calling target)
lmcorral run --probe jailbreak  # override / encoding / concealed-instruction shapes
lmcorral run --probe scope      # impossible tasks and transport boundary creep
```

Prefix matching works (`ssrf`, `jailbreak.direct_override`, etc.). Tool probes skip automatically
when the target has no tool support. The leak probe only fails when the full canary appears in the
**visible reply** — mentions in a thinking/reasoning trace do not count.

Optional canary server for SSRF (records real HTTP hits when your runtime executes tools):

```yaml
probe_server:
  host: 127.0.0.1
  port: 8765
  path: /canary/ssrf
```

### Report formats (`qwen3.5:9b` on Ollama)

`lmcorral-report.jsonl` is the full record (per-finding trial counts, run score, transcripts).
Optional human-readable exports:

- **`--docx`** — Word document (`report.docx` by default; optional path: `--docx path.docx`).
  Includes a clickable table of contents with internal links to each probe and trial.
  Headings group each probe under **Findings**, with all **Trials** nested together.
- **`--html`** — self-contained HTML (`report.html` by default; optional path: `--html path.html`).
  Sidebar, search filter, and collapsible probe/trial sections (recommended for long runs).

<!-- Image 2: word report -->
<img src="docs/LMCorral-Word1.png" alt="LMCorral word report 1" style="width: 47%; display: inline-block; margin: 0 1%;"><img src="docs/LMCorral-Word2.png" alt="LMCorral word report 2" style="width: 45%; display: inline-block; margin: 0 1%;"><br>

**Use results to fix the deployment** — tighten `num_predict`, add monitors in your own gateway,
deny-list tools, harden system prompts — so the same class of failure does not ship again. A pass
here is not a certificate; a fail is a concrete reproducer to work from.

## Add your own probes

**1. In `config.yaml`, no Python:**

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
```

`expect`: `must_contain`, `must_not_contain`, `regex`, `not_regex`, or `refused: true|false`.

**2. Python probe in your own folder** — point `probe_dirs` at it:

```python
from lmcorral.monitors import TokenBudget
from lmcorral.protocol import Outcome, Probe, Turn
from lmcorral.probes import register

@register
class MyProbe(Probe):
    id = "custom.my_probe"
    summary = "One sentence on what failure this looks for"
    owasp = "LLM10:2025 Unbounded Consumption"

    def turns(self):
        yield Turn(messages=[{"role": "user", "content": "..."}], label="attempt-1")

    def monitors(self):
        return [TokenBudget(self.limits.token_budget)]

    def judge(self, transcripts):
        if transcripts[0].aborted:
            return self.finding(Outcome.FAIL, "ran past its budget")
        return self.finding(Outcome.PASS, "stopped on its own")
```

**3. Drop a file in `lmcorral/probes/`** — discovered on the next run.

## Configuration

`config.yaml` is **optional**. After `pip install`, only **`--target`** and **`--model`**
are required. Limits, report paths, probe filters, and custom probes use **package
defaults** until you add a yaml file (in the working directory or via `--config`).

**Target** (required for `run`, one of):

- `--target URL` and `--model NAME` on the command line
- `target.url` and `target.model` in `config.yaml` (CLI overrides when both are set)

**Limits** (optional): override any key under `limits:` — omitted keys keep the package
default (same values as the example below).

**Probe server** (required for `runaway.circular_brief` and `runaway.forbidden_resolution`; also
records SSRF canary hits when tools execute for real). Set `probe_server.port` in yaml — default
is off (`port: 0`) until you enable it:

```yaml
target:
  url: http://127.0.0.1:11434
  model: "qwen3.5:9b"
  api_key: "${OPENAI_API_KEY}"   # OpenAI-compatible endpoints only
  connect_timeout_seconds: 5.0
  read_timeout_seconds: 120.0

limits:
  token_budget: 8192             # max stream chunks before cut-off (runaway probes)
  wall_clock_seconds: 300.0      # max seconds per generation
  token_gap_seconds: 20.0        # max silence between chunks
  max_tokens: 4096               # max answer length for bounded probes
  temperature: 0.7
  repetition_min_period: 3
  repetition_max_period: 256
  repetition_cycles: 5
  repetition_line_repeats: 8
  repetition_check_every: 256    # how often the repetition monitor scans the buffer

report:
  jsonl: lmcorral-report.jsonl
  docx: null                     # e.g. report.docx
  html: null                     # e.g. report.html
  max_transcript_chars: 1_000_000

probes: []                       # empty = run all; or list ids: [runaway, safety, leak]
probe_limits: {}                 # per-probe overrides, e.g. runaway.unbounded_output: {token_budget: 1200}
probe_dirs: []                   # folders of your own .py probes

probe_server:
  host: 127.0.0.1
  port: 8765
  path: /canary/ssrf

custom_probes: []                # probes without Python — see below
```

The repository ships this as an **example** `config.yaml` for tuning; copy and edit what
you need. `${VAR}` expands from the environment. Other optional CLI flags: `--probe`,
`--docx`, `--html`, `--out`, `--config`.

## Scope of control

LMCorral controls **one streaming request** from the client side. When a monitor aborts, it
**closes the connection** — you stop receiving tokens, and the server *may* stop generating if
inference is tied to that request (typical for Ollama, vLLM, and OpenAI-style chat streams).

That is not the same as stopping everything your stack might do:

- **Agent orchestrators** can keep running after a turn ends — workflows, retries, delegated agents.
- **Async or queued inference** may continue after the client disconnects.
- **Tool execution in workers** (SSH, HTTP, subprocesses) often runs **outside** the stream
  LMCorral hangs up on, unless your gateway cancels it explicitly.

The `containment.stop_button` probe checks whether **your endpoint** actually frees the slot after
abort (known caveat: some Ollama CPU-offload setups keep generating). It always reports pass or
fail — there is no skip path. It does not prove that background jobs, tool runners, or multi-agent
loops stop.

Use LMCorral for stream behaviour, leaks, refusals, and **attempted** tool calls observed in the
response. For agentic deployments, pair it with orchestrator kill switches, tool denylists, and
process sandboxing — abort at the wire is necessary but not always sufficient.

## Disclaimer

Findings are **indicators of behaviour**, not proof of safety or compromise. Heuristics
(keywords, canaries, chunk counts) miss subtle failures and false-alarm on edge cases. Read the
transcripts in the report before acting. See [Scope of control](#scope-of-control) for what abort
does and does not guarantee. This tool does not replace your own review, threat modelling, or
production controls.

## Licence

Apache 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and how to add probes or monitors.

- [Open an issue](https://github.com/gilbo123/LMCorral/issues/new)
- [View open issues](https://github.com/gilbo123/LMCorral/issues)
- [Open a pull request](https://github.com/gilbo123/LMCorral/compare)
- [View pull requests](https://github.com/gilbo123/LMCorral/pulls)
