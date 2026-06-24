"""LLM client for the cognition stages — OpenAI-compatible (MiniMax).

The 解說 script brain runs on MiniMax via its OpenAI-compatible chat endpoint
(vision through MiniMax-VL). Because the surface is OpenAI-compatible, the exact
same client also drives a self-hosted vLLM server — only ``base_url`` and ``model``
change — so swapping to an open model later is a config edit, not a code change.

Structured output is requested as a JSON object with the target schema embedded in
the instruction, then validated against our Pydantic models by the caller. A small
retry covers the occasional non-JSON reply. Keyframes are sent as base64 data-URI
``image_url`` blocks (the OpenAI vision format).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

log = logging.getLogger("yapper.llm")


class LLMError(RuntimeError):
    pass


def image_block(path: str | Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _schema_directive(schema: dict) -> str:
    return (
        "\n\n只输出一个 JSON 对象，且严格符合下面的 JSON Schema（不要输出多余的解释或 markdown 代码块）：\n"
        + json.dumps(schema, ensure_ascii=False)
    )


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        model: str = "MiniMax-M3",
        temperature: float = 0.7,
        max_output_tokens: int = 32000,
        max_retries: int = 2,
        vision: bool = True,
        max_images: int = 180,
        timeout: float = 1800.0,
        default_pre_request_hook: Callable[[], None] | None = None,
        default_on_usage: Callable[[object], None] | None = None,
    ):
        # Generous timeout: MiniMax-M3 is a reasoning model and the REDUCE pass can
        # take many minutes to generate a full script (the SDK's 600s default times out).
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.vision = vision  # whether the model accepts keyframe images
        self.max_images = max_images  # provider cap (MiniMax ≤200 images/request)
        # Optional cost-control hooks applied to every call (the platform wires the
        # budget guard here; the CLI leaves them None). Per-call args override these.
        self.default_pre_request_hook = default_pre_request_hook
        self.default_on_usage = default_on_usage

    def complete_structured(
        self,
        *,
        system_text: str,
        source_blocks: list[dict],
        instruction: str,
        schema: dict,
        thinking: str | None = None,
        pre_request_hook: Callable[[], None] | None = None,
        on_usage: Callable[[object], None] | None = None,
    ) -> tuple[dict, object]:
        """Run one structured pass; return (parsed_json, usage).

        ``source_blocks`` is the clip-tagged transcript + keyframe images (built by
        s05). ``instruction`` is the pass-specific ask; the schema directive is
        appended so the model emits conforming JSON. ``thinking`` (``"enabled"`` /
        ``"adaptive"`` / ``"disabled"``) controls MiniMax-M3 reasoning — disabling it
        on the REDUCE pass frees the output budget the model would otherwise spend on
        ``<think>`` for actual narration.

        Cost control (used by the platform; ``None`` for the CLI): ``pre_request_hook``
        is called once before the API request and may raise to abort when the spend cap
        is reached; ``on_usage`` is called with the response ``usage`` after a
        successful call so the caller can record tokens/cost.
        """
        pre_request_hook = pre_request_hook or self.default_pre_request_hook
        on_usage = on_usage or self.default_on_usage
        if pre_request_hook is not None:
            pre_request_hook()
        user_content = [
            *source_blocks,
            text_block(instruction + _schema_directive(schema)),
        ]
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]
        extra: dict = {}
        if thinking:
            # MiniMax OpenAI-compatible endpoint accepts reasoning controls via extra_body.
            extra["extra_body"] = {"thinking": {"type": thinking}}

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
                **extra,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "content_filter":
                raise LLMError("request blocked by content filter")
            text = choice.message.content or ""
            try:
                data = json.loads(_extract_json(text))
                u = resp.usage
                log.info(
                    "llm: in=%s out=%s (attempt %d)",
                    getattr(u, "prompt_tokens", "?"),
                    getattr(u, "completion_tokens", "?"),
                    attempt + 1,
                )
                if on_usage is not None:
                    on_usage(u)
                return data, u
            except (
                json.JSONDecodeError
            ) as e:  # nudge the model to fix its JSON and retry
                last_err = e
                log.warning("llm returned non-JSON (attempt %d); retrying", attempt + 1)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": "上一条不是合法 JSON。请只输出符合 schema 的 JSON 对象。",
                    }
                )

        raise LLMError(
            f"model did not return valid JSON after {self.max_retries + 1} attempts: {last_err}"
        )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reasoning-model response.

    MiniMax-M3 prepends a <think>...</think> block and may wrap JSON in a code
    fence. Strip both, then fall back to the outermost {...} span.
    """
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    t = _strip_fences(t)
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            t = t[start : end + 1]
    return t
