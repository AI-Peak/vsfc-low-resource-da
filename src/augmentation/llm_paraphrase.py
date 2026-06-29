"""Gemini paraphrasing augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
import re
import time
from typing import Any

from src.augmentation.base import TextAugmenter
from src.augmentation.prompts import build_paraphrase_prompt

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
NUMBERED_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


@dataclass(frozen=True)
class LLMParaphraseConfig:
    """Configuration for Gemini paraphrase generation."""

    model: str = "gemini-2.5-flash-lite"
    num_aug: int = 1
    temperature: float = 0.7
    max_retries: int = 3
    retry_sleep_seconds: float = 3.0
    request_sleep_seconds: float = 0.0


class GeminiParaphraser:
    """Small Gemini wrapper with robust parsing and retry logic."""

    method_name = "llm_paraphrase_raw"

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        num_aug: int = 1,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_sleep_seconds: float = 3.0,
        request_sleep_seconds: float = 0.0,
        api_key_env: str = "GEMINI_API_KEY",
        seed: int = 42,
    ) -> None:
        self.config = LLMParaphraseConfig(
            model=str(model),
            num_aug=int(num_aug),
            temperature=float(temperature),
            max_retries=int(max_retries),
            retry_sleep_seconds=float(retry_sleep_seconds),
            request_sleep_seconds=float(request_sleep_seconds),
        )
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.api_key_env = api_key_env
        self._model = None

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                f"Missing Gemini API key. Set {self.api_key_env} or GOOGLE_API_KEY."
            )
        return key

    def _client(self) -> Any:
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-generativeai is not installed. Run pip install -r requirements.txt."
                ) from exc

            genai.configure(api_key=self._api_key())
            self._model = genai.GenerativeModel(self.config.model)
        return self._model

    @staticmethod
    def _clean_candidate(text: str) -> str:
        text = TextAugmenter.normalize_text(text)
        text = NUMBERED_LINE_RE.sub("", text).strip()
        text = text.strip("\"'` ")
        return TextAugmenter.normalize_text(text)

    @classmethod
    def parse_response(cls, raw_text: str) -> list[str]:
        """Parse a Gemini response into paraphrase strings."""
        text = raw_text.strip()
        fence = JSON_FENCE_RE.search(text)
        if fence:
            text = fence.group(1).strip()

        json_candidates = [text]
        if "[" in text and "]" in text:
            json_candidates.append(text[text.find("[") : text.rfind("]") + 1])

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ("paraphrases", "sentences", "results"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return [cls._clean_candidate(str(item)) for item in value]
            if isinstance(parsed, list):
                return [cls._clean_candidate(str(item)) for item in parsed]

        lines = [
            cls._clean_candidate(line)
            for line in text.splitlines()
            if cls._clean_candidate(line)
        ]
        return lines

    @staticmethod
    def deduplicate(sentences: list[str]) -> list[str]:
        """Deduplicate while preserving order."""
        seen: set[str] = set()
        output: list[str] = []
        for sentence in sentences:
            normalized = TextAugmenter.normalize_text(sentence)
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
        return output

    def paraphrase(self, sentence: str, label: int, num_aug: int | None = None) -> list[str]:
        """Generate label-preserving paraphrases for one sentence."""
        resolved_num_aug = int(num_aug if num_aug is not None else self.config.num_aug)
        prompt = build_paraphrase_prompt(sentence, label=label, num_aug=resolved_num_aug)
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client().generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.config.temperature,
                    },
                )
                raw_text = getattr(response, "text", "") or ""
                candidates = self.deduplicate(self.parse_response(raw_text))
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate and candidate != TextAugmenter.normalize_text(sentence)
                ]
                if candidates:
                    if self.config.request_sleep_seconds > 0:
                        time.sleep(self.config.request_sleep_seconds)
                    return candidates[:resolved_num_aug]
                last_error = ValueError(f"Empty paraphrase response: {raw_text[:200]}")
            except Exception as exc:  # pragma: no cover - exercised on Kaggle/API side
                last_error = exc

            sleep_seconds = self.config.retry_sleep_seconds * attempt
            time.sleep(sleep_seconds)

        raise RuntimeError(f"Gemini paraphrase failed after retries: {last_error}")
