"""OpenAI Chat Completions client with tool calls, safe errors and usage traces."""
import copy
import json
import time
import urllib.error
import urllib.request


class ChatAPIError(RuntimeError):
    """Safe provider error without headers, credentials or raw response bodies."""


def estimate_cost(usage, prices):
    if not isinstance(usage, dict) or not all(k in usage for k in ("prompt_tokens", "completion_tokens")):
        return None
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return ((usage["prompt_tokens"] - cached) * prices["input"]
            + cached * prices["cached_input"]
            + usage["completion_tokens"] * prices["output"]) / 1_000_000


class OpenAIController:
    def __init__(self, api_key, config):
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing from .env")
        if config["provider"] != "OpenAI" or config["model"] != "gpt-4o":
            raise ValueError("This controller is configured for OpenAI gpt-4o")
        self._key, self.config = api_key, dict(config)
        self.model_name, self.calls = config["model"], []

    def complete(self, messages, tools):
        payload = {"model": self.model_name, "messages": messages, "tools": tools,
                   "tool_choice": "required", "parallel_tool_calls": False,
                   "temperature": self.config["temperature"],
                   "max_completion_tokens": self.config["max_completion_tokens"]}
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, 3):
            event = {"attempt": attempt, "request": copy.deepcopy(payload)}
            self.calls.append(event)
            started = time.perf_counter()
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=request_body,
                headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                if not result.get("choices"):
                    raise ChatAPIError("OpenAI returned no choices")
                choice = result["choices"][0]
                message = choice["message"]
                # Retain public output/tool calls; never request or save hidden reasoning.
                message = {k: message[k] for k in ("role", "content", "tool_calls", "refusal") if k in message}
                event.update({"response_id": result.get("id"), "model_returned": result.get("model"),
                              "usage": result.get("usage"), "finish_reason": choice.get("finish_reason"),
                              "message": message,
                              "estimated_cost_usd": estimate_cost(result.get("usage"), self.config["prices_per_million_tokens"])})
                if choice.get("finish_reason") != "tool_calls":
                    raise ChatAPIError("OpenAI did not return a completed tool call")
                return message
            except urllib.error.HTTPError as error:
                try:
                    code = json.load(error).get("error", {}).get("code")
                except (ValueError, AttributeError):
                    code = None
                event["error"] = {"type": "HTTPError", "http_status": error.code}
                if code in {"insufficient_quota", "invalid_api_key", "model_not_found", "rate_limit_exceeded"}:
                    event["error"]["provider_code"] = code
                retry = error.code in {429, 500, 502, 503, 504} and code != "insufficient_quota"
                if not retry or attempt == 2:
                    raise ChatAPIError(f"OpenAI HTTP {error.code}; check API access, quota or service status") from None
            except (urllib.error.URLError, TimeoutError):
                event["error"] = {"type": "NetworkError"}
                if attempt == 2:
                    raise ChatAPIError("OpenAI network request failed") from None
            except (ValueError, KeyError, TypeError, ChatAPIError):
                event["error"] = {"type": "InvalidResponse"}
                raise ChatAPIError("OpenAI returned an invalid or incomplete tool response") from None
            finally:
                event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            time.sleep(2 ** attempt)
