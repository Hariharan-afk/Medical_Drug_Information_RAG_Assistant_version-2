#generator.py
"""LLM adapter layer for the medical RAG system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when fallback used
    AutoModelForSeq2SeqLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from config import (
    LLM_API_KEY,
    LLM_API_KEY_HEADER,
    LLM_BASE_URL,
    LLM_PROVIDER,
    MISTRAL_MODEL,
    TIMEOUT_SECONDS,
)
from utils import setup_logger


logger = setup_logger("generator")


class GeneratorError(Exception):
    """Raised when generation fails."""


class BaseClient:
    name = "base"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        raise NotImplementedError


class MistralOpenAICompatClient(BaseClient):
    name = "mistral-openai-compat"

    def __init__(self, base_url: str, api_key: str, header: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.header = header
        self.timeout = TIMEOUT_SECONDS
        self.session = httpx.Client(timeout=self.timeout)

        print(f"DEBUG MistralOpenAICompatClient:")
        print(f"  base_url = '{self.base_url}'")
        print(f"  api_key = '{self.api_key[:20]}...' (truncated)")
        print(f"  is_configured = {self.is_configured}")

    @property
    def is_configured(self) -> bool:
        result = "YOUR-LLM-ENDPOINT" not in self.base_url
        print(f"DEBUG is_configured check: 'YOUR-LLM-ENDPOINT' not in '{self.base_url}' = {result}")
        return result

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.upper() != "YOUR-API-KEY":
            headers[self.header] = f"Bearer {self.api_key}"

        payload = {
            "model": MISTRAL_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GeneratorError(f"OpenAI-compatible request failed: {exc}") from exc

        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise GeneratorError("No choices returned from OpenAI-compatible endpoint")
        return choices[0]["message"]["content"].strip()


class MistralOllamaClient(BaseClient):
    name = "mistral-ollama"

    def __init__(self, base_url: str, api_key: str, header: str) -> None:
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.api_key = api_key
        self.header = header
        self.timeout = TIMEOUT_SECONDS
        self.session = httpx.Client(timeout=self.timeout)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.upper() != "YOUR-API-KEY":
            headers[self.header] = self.api_key

        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GeneratorError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        message = data.get("message", {})
        content = message.get("content") or data.get("response")
        if not content:
            raise GeneratorError("Ollama response missing content")
        return content.strip()


class FlanT5LocalClient(BaseClient):
    name = "flan-t5-base"

    def __init__(self) -> None:
        self.model_name = "google/flan-t5-base"
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise GeneratorError(
                "Transformers is required for the FLAN-T5 fallback. Install the 'transformers' package."
            )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> str:
        self._ensure_loaded()
        prompt = f"{system_prompt}\n\n{user_prompt}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self._model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(0.1, temperature),
            max_new_tokens=max_tokens,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()


@dataclass
class GenerationResult:
    answer: str
    backend: str
    used_fallback: bool


class GeneratorRouter:
    def __init__(self) -> None:
        self.primary: Optional[BaseClient] = None
        if LLM_PROVIDER == "openai_compat":
            candidate = MistralOpenAICompatClient(LLM_BASE_URL, LLM_API_KEY, LLM_API_KEY_HEADER)
            if candidate.is_configured:
                self.primary = candidate
            else:
                logger.warning(
                    "LLM_BASE_URL still contains placeholder; OpenAI-compatible backend disabled."
                )
        elif LLM_PROVIDER == "ollama":
            self.primary = MistralOllamaClient(LLM_BASE_URL, LLM_API_KEY, LLM_API_KEY_HEADER)

        self.fallback = FlanT5LocalClient()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        clients = [self.primary, self.fallback] if self.primary else [self.fallback]
        for client in filter(None, clients):
            try:
                answer = client.generate(system_prompt, user_prompt, temperature, max_tokens)
                return GenerationResult(
                    answer=answer,
                    backend=client.name,
                    used_fallback=client is self.fallback and self.primary is not None,
                )
            except GeneratorError as exc:
                logger.warning("Generation failed with %s: %s", client.name, exc)
                continue
        raise GeneratorError("All generation backends failed")


__all__ = [
    "GeneratorRouter",
    "GenerationResult",
    "GeneratorError",
    "MistralOpenAICompatClient",
    "MistralOllamaClient",
    "FlanT5LocalClient",
]

def __init__(self) -> None:
    print(f"DEBUG GeneratorRouter init:")
    print(f"  LLM_PROVIDER = '{LLM_PROVIDER}'")
    print(f"  LLM_BASE_URL = '{LLM_BASE_URL}'")
    print(f"  LLM_API_KEY = '{LLM_API_KEY[:20] if LLM_API_KEY else None}...'")
    
    self.primary: Optional[BaseClient] = None
    if LLM_PROVIDER == "openai_compat":
        candidate = MistralOpenAICompatClient(LLM_BASE_URL, LLM_API_KEY, LLM_API_KEY_HEADER)
        if candidate.is_configured:
            self.primary = candidate
            print("DEBUG: Primary client set to MistralOpenAICompatClient")
        else:
            logger.warning(
                "LLM_BASE_URL still contains placeholder; OpenAI-compatible backend disabled."
            )
            print("DEBUG: Primary client NOT set (placeholder detected)")
    elif LLM_PROVIDER == "ollama":
        self.primary = MistralOllamaClient(LLM_BASE_URL, LLM_API_KEY, LLM_API_KEY_HEADER)
        print("DEBUG: Primary client set to MistralOllamaClient")

    self.fallback = FlanT5LocalClient()