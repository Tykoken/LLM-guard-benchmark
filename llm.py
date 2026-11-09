

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from typing import Any

from openai import OpenAI

from config import BASE_URL, REQUEST_TIMEOUT


def extract_json(text: str | None) -> Any | None:

    if not text:
        return None
    text = re.sub(r"<\|[^>]*\|>", "", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    candidates = []
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except Exception:
        return None


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


class LLMClient:
    def __init__(self, settings: Any):
        self.settings = settings
        api_key = settings.key.read_text(encoding="utf-8").strip()
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0}

    def _cache_path(self, key: str):
        return self.settings.cache_dir / key[:2] / f"{key}.json"

    @staticmethod
    def _key(
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        cache_salt: str = "",
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
                "cache_salt": cache_salt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        json_mode: bool = True,
        tag: str = "",
        cache_salt: str = "",
    ) -> dict[str, Any]:
        key = self._key(model, messages, max_tokens, temperature, json_mode, cache_salt)
        path = self._cache_path(key)
        if path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                record["cached"] = True
                record["latency_ms"] = 0
                with self._lock:
                    self.stats["cache_hits"] += 1
                return record
            except Exception:
                pass

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        effective_max_tokens = max_tokens
        for attempt in range(6):
            try:
                kwargs["max_tokens"] = effective_max_tokens
                started = time.time()
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = int((time.time() - started) * 1000)
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length" and effective_max_tokens < 6000 and attempt < 2:
                    effective_max_tokens = min(6000, int(effective_max_tokens * 1.6))
                    continue
                usage = usage_dict(response)
                record = {
                    "model": model,
                    "text": response.choices[0].message.content or "",
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "cached": False,
                    "tag": tag,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
                with self._lock:
                    self.stats["calls"] += 1
                return record
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                text = str(exc)
                if json_mode and status == 400 and "response_format" in text:
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or any(
                    marker in text.lower()
                    for marker in ("timeout", "timed out", "connection", "temporarily", "overloaded")
                )
                if not retryable or attempt == 5:
                    break
                time.sleep(min(2**attempt, 30) + random.random())
        with self._lock:
            self.stats["errors"] += 1
        raise RuntimeError(f"DeepSeek call failed after retries: {last_error}") from last_error

    def call_json(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tag: str = "",
        attempts: int = 2,
        cache_salt: str = "",
    ) -> tuple[Any | None, dict[str, Any]]:

        current = list(messages)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_latency = 0
        last_record: dict[str, Any] = {}
        for attempt in range(attempts):
            record = self.call(
                model=model,
                messages=current,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
                tag=tag,
                cache_salt=cache_salt,
            )
            last_record = record
            for key in total_usage:
                total_usage[key] += record.get("usage", {}).get(key, 0)
            if not record.get("cached"):
                total_latency += record.get("latency_ms", 0)
            parsed = extract_json(record.get("text"))
            if parsed is not None:
                meta = dict(record)
                meta["usage"] = total_usage
                meta["latency_ms"] = total_latency
                meta["attempts"] = attempt + 1
                return parsed, meta
            current = list(messages) + [
                {"role": "assistant", "content": record.get("text", "")},
                {
                    "role": "user",
                    "content": "Your previous reply was not valid JSON. Return only the JSON object, with no markdown, prose, or extra text.",
                },
            ]
        meta = dict(last_record)
        meta["usage"] = total_usage
        meta["latency_ms"] = total_latency
        meta["attempts"] = attempts
        meta["parse_error"] = True
        return None, meta
