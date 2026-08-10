<!-- Image 1: logo -->
![LMCorral logo](docs/LMCorral-logo.png)

# LMCorral

**Corral your language models.** Stress-test them outside production, and cut the stream when they bolt.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Issues](https://img.shields.io/github/issues/gilbo123/LMCorral)](https://github.com/gilbo123/LMCorral/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/gilbo123/LMCorral)](https://github.com/gilbo123/LMCorral/pulls)

### 1. Install LMCorral on your laptop or server

```bash
git clone https://github.com/gilbo123/LMCorral.git && cd LMCorral
uv sync                    # creates .venv and installs deps (recommended)
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Point LMCorral at your model server

Pass the endpoint on the command line (works from any directory after `pip install`):

```bash
lmcorral run --target http://127.0.0.1:11434 --model qwen3.6:latest
```

Or put defaults in `config.yaml` in the directory where you run the tool:

```yaml
target:
  url: http://127.0.0.1:11434
  model: qwen3.6:latest
```

CLI `--target` and `--model` override the config file when both are set.

### 3. Run

```bash
lmcorral run --target http://127.0.0.1:11434 --model qwen3.6:latest

# with config.yaml for limits/reports; override target on the CLI if you prefer
lmcorral run --verbose --probe ssrf --docx report.docx
```

### Table output example (`qwen3.6:latest` on Ollama)

<!-- Image 3: table report -->
![LMCorral table report](docs/LMCorral-Table.png)


## What it is

**LMCorral** sits in front of a model server you already run (Ollama,
vLLM, OpenAI-compatible APIs) and watches the response **as it streams**. Monitors can hang up
mid-generation — runaway output, repetition, leaked secrets, denied tool calls — without copying
the model or rebuilding your stack.

That is restraint at the wire, not isolation of the weights. Malicious artifacts at load time are
a different problem; this tool targets failures that show up once the endpoint is already
answering.

## Usage

```bash
lmcorral run --verbose
lmcorral report lmcorral-report.jsonl
python -m lmcorral run         # module form; works with venv active (pip or uv)
```

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
when the target has no tool support.

Optional canary server for SSRF (records real HTTP hits when your runtime executes tools):

```yaml
probe_server:
  host: 127.0.0.1
  port: 8765
  path: /canary/ssrf
```

### Word report example (`qwen3.6:latest` on Ollama)

`lmcorral-report.jsonl` (detail) and optional `--docx report.docx`.

<!-- Image 2: word report -->
<img src="docs/LMCorral-Word1.png" alt="LMCorral word report 1" style="width: 45%; display: inline-block; margin: 0 1%;"><img src="docs/LMCorral-Word2.png" alt="LMCorral word report 2" style="width: 45%; display: inline-block; margin: 0 1%;"><br>

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

`config.yaml` is **optional**. Use it for limits, reports, custom probes, and default
`target` settings. The file is read from the current directory (or `--config path`) when
present.

**Target** (required for `run`, one of):

- `--target URL` and `--model NAME` on the command line
- `target.url` and `target.model` in `config.yaml`

CLI flags override the config file. This matches common CLI tools (`curl`, `kubectl`,
database clients): point at a server per invocation without installing config files.

```yaml
target:
  url: http://127.0.0.1:11434
  model: qwen3.6:latest
  api_key: "${OPENAI_API_KEY}"   # OpenAI-compatible endpoints only

limits:
  token_budget: 600
  wall_clock_seconds: 45.0

report:
  jsonl: lmcorral-report.jsonl
  docx: null
```

`${VAR}` expands from the environment. Other optional CLI flags: `--probe`, `--docx`,
`--out`, `--config`.

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
abort (known caveat: some Ollama CPU-offload setups keep generating). It does not prove that
background jobs, tool runners, or multi-agent loops stop.

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
