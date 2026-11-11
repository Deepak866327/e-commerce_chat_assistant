"""Bounded local Ollama HTTP calls with schema validation and limited retries."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError


class ModelUnavailable(RuntimeError):
    pass


class InvalidModelOutput(ValueError):
    pass


class OllamaClient:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport or self._request
        url = urlparse(settings.ollama_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Ollama URL must be HTTP(S) without embedded credentials")
        self.calls = []

    def model_digest(self, model):
        request = Request(self.settings.ollama_url.rstrip("/") + "/api/tags")
        try:
            with urlopen(request, timeout=min(10, self.settings.model_timeout)) as response:
                data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ModelUnavailable("Model catalog exceeds the response limit")
            catalog = json.loads(data)
            candidates = {model, model + ":latest"} if ":" not in model else {model}
            for record in catalog.get("models", []):
                if record.get("name") in candidates and record.get("digest"):
                    return record["digest"]
            raise ModelUnavailable("Required Ollama model is not installed")
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelUnavailable("Could not verify the local Ollama model identity") from exc

    def _request(self, payload):
        request = Request(self.settings.ollama_url.rstrip("/") + "/api/chat",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.settings.model_timeout) as response:
                data = response.read(2 * 1024 * 1024 + 1)
                if len(data) > 2 * 1024 * 1024:
                    raise InvalidModelOutput("Ollama response exceeds limit")
                return json.loads(data)
        except HTTPError as exc:
            raise ModelUnavailable(f"Ollama returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ModelUnavailable("Ollama is unavailable or timed out") from exc

    def structured(self, model, messages, schema, retries=1, num_ctx=4096, max_output_tokens=1024):
        if not 1 <= max_output_tokens <= 8192 or num_ctx < 512:
            raise ValueError("Invalid structured-output token budget")
        conversation = list(messages)
        last_error = None
        for attempt in range(retries + 1):
            payload = {"model": model, "messages": conversation, "stream": False,
                       "format": schema.model_json_schema(), "keep_alive": "5m",
                       "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": max_output_tokens}}
            started = time.perf_counter()
            try:
                response = self.transport(payload)
                content = response["message"]["content"]
                result = schema.model_validate_json(content)
                self.calls.append({"model": model, "seconds": time.perf_counter() - started,
                                   "eval_count": response.get("eval_count"), "ok": True})
                return result
            except ModelUnavailable:
                self.calls.append({"model": model, "seconds": time.perf_counter() - started, "ok": False})
                raise
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                last_error = exc
                self.calls.append({"model": model, "seconds": time.perf_counter() - started, "ok": False})
                if attempt < retries:
                    conversation = list(messages) + [{"role": "user", "content":
                        "The previous output did not match the schema. Return one valid JSON object conforming to the supplied schema."}]
        raise InvalidModelOutput("Model returned invalid structured output after bounded retries") from last_error
