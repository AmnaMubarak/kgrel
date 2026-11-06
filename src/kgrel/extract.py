"""Model clients behind a single interface.

Backends supported: `openai`, `groq`, `mock`. New backends subclass `BaseClient`
and register themselves. The mock backend returns deterministic triples so the
pilot smoke test runs with no API keys.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .parse import parse_triples
from .prompt import render_messages
from .schema import GoldItem, Ontology, Prediction, Triple


@dataclass
class ModelSpec:
    name: str                   # internal name used in results
    backend: str                # "openai" | "groq" | "mock"
    api_model_id: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512


class BaseClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def call(self, messages: List[Dict[str, str]]) -> str:  # returns raw string
        raise NotImplementedError


class MockClient(BaseClient):
    """Deterministic client used for dry runs. Emits the first gold triple
    (if available via the message) or an empty list.
    """
    def call(self, messages: List[Dict[str, str]]) -> str:
        # We cannot peek at gold here, so just emit an empty list — the pilot
        # smoke test uses a separate helper that fakes model outputs from gold.
        return json.dumps({"triples": []})


class OpenAIClient(BaseClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai` or use backend=mock."
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = OpenAI(api_key=api_key)

    def call(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.spec.api_model_id or self.spec.name,
            messages=messages,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class GroqClient(BaseClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from groq import Groq  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "groq package not installed. `pip install groq` or use backend=mock."
            ) from e
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        self._client = Groq(api_key=api_key)
        # Some models on Groq do not accept the strict `response_format=json_object`
        # and will return empty content with a validation error. For those we fall
        # back to a strong prompt-level JSON instruction (already in SYSTEM).
        #   - gemma*            : documented limitation
        #   - gpt-oss-20b       : small OSS variant chokes on the schema at low temps
        name = (spec.api_model_id or spec.name).lower()
        _no_json_mode = ("gemma", "gpt-oss-20b")
        self._supports_json_mode = not any(tok in name for tok in _no_json_mode)

    def call(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.spec.api_model_id or self.spec.name,
            messages=messages,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
        )
        if self._supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Simple backoff on 429 / transient errors. Groq free tier is generous
        # but bursty extraction (900 calls / 3 models) can trip the per-minute
        # limit; a short sleep is cheaper than losing a run.
        delay = 2.0
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                msg = str(e).lower()
                transient = ("rate" in msg or "429" in msg
                             or "timeout" in msg or "502" in msg or "503" in msg)
                if not transient or attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2


_REGISTRY: Dict[str, type] = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "mock": MockClient,
}


def make_client(spec: ModelSpec) -> BaseClient:
    cls = _REGISTRY.get(spec.backend)
    if cls is None:
        raise ValueError(f"unknown backend: {spec.backend}")
    return cls(spec)


def extract_one(client: BaseClient, item: GoldItem, ontology: Ontology,
                run_id: str = "default") -> Prediction:
    msgs = render_messages(item, ontology)
    t0 = time.time()
    try:
        raw = client.call(msgs)
    except Exception as e:
        raw = f"__ERROR__: {e}"
    latency = time.time() - t0
    triples, ok = parse_triples(raw)
    return Prediction(
        sent_id=item.sent_id,
        model=client.spec.name,
        run_id=run_id,
        triples=triples,
        raw_output=raw,
        parse_ok=ok,
        latency_s=latency,
    )
