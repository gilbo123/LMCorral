"""Model endpoints.

A target does one interesting thing: it lets a monitor hang up the phone.

Both Ollama and llama.cpp/vLLM tie generation to the lifetime of the HTTP
request, so closing the connection cancels the work server-side. That makes an
ordinary reverse-proxy position into a kill switch, with no cooperation needed
from the server and no change to how the model was started.

The caveat is documented and load-bearing: ollama/ollama#2876 reports that when
layers are offloaded to CPU, generation can survive the disconnect. Truncating
the *stream* is not the same as stopping the *work*, so `settling_delay` below
measures the difference rather than assuming it away.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import httpx

from .protocol import Action, Monitor, StreamView, Target, Transcript, Turn


class TargetError(RuntimeError):
    pass


class OllamaTarget(Target):
    """Ollama's native `/api/chat`, streamed as newline-delimited JSON."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/api/chat").removesuffix("/api")
        self.model = model
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    # -- discovery ---------------------------------------------------------- #

    def health(self) -> tuple[bool, str]:
        try:
            tags = self._client.get("/api/tags").json()
        except httpx.HTTPError as exc:
            return False, f"cannot reach {self.base_url}: {exc}"
        names = [m["name"] for m in tags.get("models", [])]
        if not names:
            return False, f"{self.base_url} has no models pulled"
        if not self.model:
            self.model = names[0]
        elif self.model not in names and f"{self.model}:latest" not in names:
            return False, f"model {self.model!r} not found; available: {', '.join(names)}"
        return True, f"{self.model} on {self.base_url}"

    def capabilities(self) -> set[str]:
        try:
            info = self._client.post("/api/show", json={"model": self.model}).json()
        except httpx.HTTPError:
            return set()
        return set(info.get("capabilities", []))

    def loaded(self) -> list[dict[str, Any]]:
        """Models currently resident, from `/api/ps`."""
        try:
            return self._client.get("/api/ps").json().get("models", [])
        except httpx.HTTPError:
            return []

    # -- generation --------------------------------------------------------- #

    def stream(self, turn: Turn, monitors: Sequence[Monitor]) -> Transcript:
        for monitor in monitors:
            monitor.reset()

        options = dict(turn.options)
        if turn.uncapped:
            # -1 is Ollama's "generate until the model stops". A probe testing
            # for runaway output must not be rescued by a server-side ceiling,
            # or a pass would prove nothing about our own gate.
            options["num_predict"] = -1

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turn.messages,
            "stream": True,
            "options": options,
        }
        if turn.tools:
            payload["tools"] = turn.tools

        transcript = Transcript(label=turn.label)
        started = time.monotonic()

        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    transcript.error = f"HTTP {response.status_code}: {response.text[:400]}"
                    transcript.elapsed_s = time.monotonic() - started
                    return transcript

                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = chunk.get("message") or {}
                    delta = message.get("content") or ""
                    thinking = message.get("thinking") or ""
                    calls = message.get("tool_calls") or []

                    transcript.chunks += 1
                    transcript.text += delta
                    transcript.reasoning += thinking
                    transcript.tool_calls.extend(calls)
                    if transcript.ttft_s is None and (delta or thinking or calls):
                        transcript.ttft_s = time.monotonic() - started

                    if chunk.get("done"):
                        transcript.server = {
                            k: v for k, v in chunk.items() if k not in {"message", "model"}
                        }

                    view = StreamView(
                        # Reasoning traces are where runaway loops usually live,
                        # so monitors see them as ordinary output.
                        delta=delta or thinking,
                        text=transcript.text or transcript.reasoning,
                        index=transcript.chunks,
                        elapsed_s=time.monotonic() - started,
                        tool_calls=transcript.tool_calls,
                        done=bool(chunk.get("done")),
                    )

                    stop = False
                    for monitor in monitors:
                        signal = monitor.observe(view)
                        if signal is None:
                            continue
                        transcript.signals.append(signal)
                        if signal.action is Action.ABORT:
                            transcript.aborted = True
                            stop = True
                    if stop:
                        # Leaving the context manager unread closes the socket,
                        # which cancels the request context server-side.
                        break

        except httpx.ReadTimeout:
            transcript.error = "read timeout"
        except httpx.HTTPError as exc:
            transcript.error = f"{type(exc).__name__}: {exc}"

        transcript.elapsed_s = time.monotonic() - started
        return transcript

    # -- did the hang-up actually work? ------------------------------------- #

    def settling_delay(self, *, timeout: float = 30.0) -> tuple[float, str]:
        """Time a one-token request immediately after an abort.

        Ollama serialises requests to a model, so if the aborted generation is
        still running this request queues behind it. A delay far above the
        model's normal time-to-first-token means the stream stopped but the GPU
        did not.
        """
        turn = Turn(
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            options={"num_predict": 1, "temperature": 0},
        )
        started = time.monotonic()
        try:
            with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": turn.messages,
                    "stream": True,
                    "options": turn.options,
                },
                timeout=httpx.Timeout(timeout, connect=5.0),
            ) as response:
                for line in response.iter_lines():
                    if line.strip():
                        return time.monotonic() - started, ""
        except httpx.HTTPError as exc:
            return time.monotonic() - started, f"{type(exc).__name__}: {exc}"
        return time.monotonic() - started, "stream closed without a token"

    def close(self) -> None:
        self._client.close()


class OpenAITarget(Target):
    """Any OpenAI-compatible `/v1/chat/completions` endpoint: vLLM, llama.cpp,
    LM Studio, TGI, or Ollama's compatibility shim."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/chat/completions")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    def health(self) -> tuple[bool, str]:
        try:
            models = self._client.get("/models").json()
        except httpx.HTTPError as exc:
            return False, f"cannot reach {self.base_url}: {exc}"
        names = [m["id"] for m in models.get("data", [])]
        if not self.model and names:
            self.model = names[0]
        return True, f"{self.model} on {self.base_url}"

    def stream(self, turn: Turn, monitors: Sequence[Monitor]) -> Transcript:
        for monitor in monitors:
            monitor.reset()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turn.messages,
            "stream": True,
        }
        for key, value in turn.options.items():
            payload["max_tokens" if key == "num_predict" else key] = value
        if turn.uncapped:
            payload.pop("max_tokens", None)
        if turn.tools:
            payload["tools"] = turn.tools

        transcript = Transcript(label=turn.label)
        started = time.monotonic()

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    transcript.error = f"HTTP {response.status_code}: {response.text[:400]}"
                    transcript.elapsed_s = time.monotonic() - started
                    return transcript

                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or [{}]
                    delta_obj = choices[0].get("delta") or {}
                    delta = delta_obj.get("content") or ""
                    thinking = delta_obj.get("reasoning_content") or ""
                    calls = delta_obj.get("tool_calls") or []

                    transcript.chunks += 1
                    transcript.text += delta
                    transcript.reasoning += thinking
                    transcript.tool_calls.extend(calls)
                    if transcript.ttft_s is None and (delta or thinking or calls):
                        transcript.ttft_s = time.monotonic() - started

                    view = StreamView(
                        delta=delta or thinking,
                        text=transcript.text or transcript.reasoning,
                        index=transcript.chunks,
                        elapsed_s=time.monotonic() - started,
                        tool_calls=transcript.tool_calls,
                        done=bool(choices[0].get("finish_reason")),
                    )

                    stop = False
                    for monitor in monitors:
                        signal = monitor.observe(view)
                        if signal is None:
                            continue
                        transcript.signals.append(signal)
                        if signal.action is Action.ABORT:
                            transcript.aborted = True
                            stop = True
                    if stop:
                        break

        except httpx.ReadTimeout:
            transcript.error = "read timeout"
        except httpx.HTTPError as exc:
            transcript.error = f"{type(exc).__name__}: {exc}"

        transcript.elapsed_s = time.monotonic() - started
        return transcript

    def settling_delay(self, *, timeout: float = 30.0) -> tuple[float, str]:
        started = time.monotonic()
        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "stream": True,
                    "max_tokens": 1,
                },
                timeout=httpx.Timeout(timeout, connect=5.0),
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        return time.monotonic() - started, ""
        except httpx.HTTPError as exc:
            return time.monotonic() - started, f"{type(exc).__name__}: {exc}"
        return time.monotonic() - started, "stream closed without a token"

    def close(self) -> None:
        self._client.close()


def build_target(url: str, model: str, *, api_key: str = "", read_timeout: float = 120.0) -> Target:
    """Pick a target from the shape of the URL, so `--target` is all anyone types."""
    if "/v1" in url or "openai" in url:
        return OpenAITarget(url, model, api_key=api_key, read_timeout=read_timeout)
    return OllamaTarget(url, model, read_timeout=read_timeout)
