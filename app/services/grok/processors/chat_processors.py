"""
聊天响应处理器
"""

import asyncio
import uuid
import re
from typing import Any, AsyncGenerator, AsyncIterable

import orjson
from curl_cffi.requests.errors import RequestsError

from app.core.config import get_config
from app.core.logger import logger
from app.core.exceptions import UpstreamException
from app.services.grok.tool_calls import ToolCallPlanner
from .base import (
    BaseProcessor,
    StreamIdleTimeoutError,
    _with_idle_timeout,
    _normalize_stream_line,
    _collect_image_urls,
    _is_http2_stream_error,
)


def extract_tool_text(raw: str, rollout_id: str = "") -> str:
    """从 xai:tool_usage_card 标签中提取工具调用信息并格式化为可读文本。

    支持的工具类型:
    - web_search -> [WebSearch]
    - search_images -> [SearchImage]
    - chatroom_send -> [AgentThink]

    Args:
        raw: 原始 XML 内容（包含 xai:tool_usage_card 标签）
        rollout_id: 可选的 rollout 标识符，用于前缀标签

    Returns:
        格式化后的工具调用文本，如 "[rolloutId][WebSearch] query_text"
    """
    if not raw:
        return ""
    name_match = re.search(
        r"<xai:tool_name>(.*?)</xai:tool_name>", raw, flags=re.DOTALL
    )
    args_match = re.search(
        r"<xai:tool_args>(.*?)</xai:tool_args>", raw, flags=re.DOTALL
    )

    name = name_match.group(1) if name_match else ""
    if name:
        name = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", name, flags=re.DOTALL).strip()

    args = args_match.group(1) if args_match else ""
    if args:
        args = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", args, flags=re.DOTALL).strip()

    payload = None
    if args:
        try:
            payload = orjson.loads(args)
        except orjson.JSONDecodeError:
            payload = None

    label = name
    text = args
    prefix = f"[{rollout_id}]" if rollout_id else ""

    if name == "web_search":
        label = f"{prefix}[WebSearch]"
        if isinstance(payload, dict):
            text = payload.get("query") or payload.get("q") or ""
    elif name == "search_images":
        label = f"{prefix}[SearchImage]"
        if isinstance(payload, dict):
            text = (
                payload.get("image_description")
                or payload.get("description")
                or payload.get("query")
                or ""
            )
    elif name == "chatroom_send":
        label = f"{prefix}[AgentThink]"
        if isinstance(payload, dict):
            text = payload.get("message") or ""

    if label and text:
        return f"{label} {text}".strip()
    if label:
        return label
    if text:
        return text
    # Fallback: 去除标签保留原始文本
    return re.sub(r"<[^>]+>", "", raw, flags=re.DOTALL).strip()


class StreamProcessor(BaseProcessor):
    """流式响应处理器"""

    def __init__(
        self,
        model: str,
        token: str = "",
        think: bool = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        super().__init__(model, token)
        self.response_id: str = None
        self.fingerprint: str = ""
        self.rollout_id: str = ""
        self.think_opened: bool = False
        self.image_think_active: bool = False
        self._last_thinking_state: bool = False
        self.answer_text: str = ""
        self.role_sent: bool = False
        self.filter_tags = get_config("chat.filter_tags")
        self.image_format = get_config("app.image_format")
        self._tag_buffer: str = ""
        self._in_filter_tag: bool = False
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.tool_call_simulation_enabled = bool(
            get_config("chat.tool_call_simulation_enabled", True)
        )
        self.tool_call_prefix = get_config("chat.tool_call_prefix", "<|tool_call|>")
        self.tool_call_strict = bool(get_config("chat.tool_call_strict", True))
        self._tool_planner = ToolCallPlanner(
            prefix=self.tool_call_prefix,
            strict=self.tool_call_strict,
        )
        self._tool_branch_enabled = (
            self.tool_call_simulation_enabled
            and bool(self.tools)
            and self.tool_choice != "none"
        )
        self._tool_mode_determined = not self._tool_branch_enabled
        self._tool_mode_enabled = False
        self._tool_gate_buffer = ""
        self._tool_raw_buffer = ""

        # 工具调用卡片提取相关
        self.tool_usage_enabled: bool = "xai:tool_usage_card" in (self.filter_tags or [])
        self._tool_usage_opened: bool = False
        self._tool_usage_buffer: str = ""

        if think is None:
            self.show_think = get_config("chat.thinking")
        else:
            self.show_think = think

    def _filter_tool_card(self, token: str) -> str:
        """处理 xai:tool_usage_card 标签：提取工具调用信息而非简单删除。

        支持跨 token 边界的标签匹配，将完整的 tool_usage_card 内容
        通过 extract_tool_text() 转换为可读的工具调用文本。
        """
        start_tag = "<xai:tool_usage_card"
        end_tag = "</xai:tool_usage_card>"

        if self._tool_usage_opened:
            self._tool_usage_buffer += token
            end_idx = self._tool_usage_buffer.find(end_tag)
            if end_idx == -1:
                return ""
            end_pos = end_idx + len(end_tag)
            raw_card = self._tool_usage_buffer[:end_pos]
            rest = self._tool_usage_buffer[end_pos:]
            self._tool_usage_opened = False
            self._tool_usage_buffer = ""
            line = extract_tool_text(raw_card, self.rollout_id)
            result = f"{line}\n" if line else ""
            if rest:
                result += self._filter_tool_card(rest)
            return result

        output_parts = []
        rest = token
        while rest:
            start_idx = rest.find(start_tag)
            if start_idx == -1:
                output_parts.append(rest)
                break

            if start_idx > 0:
                output_parts.append(rest[:start_idx])
            rest = rest[start_idx:]

            end_idx = rest.find(end_tag, len(start_tag))
            if end_idx == -1:
                self._tool_usage_opened = True
                self._tool_usage_buffer = rest
                break

            end_pos = end_idx + len(end_tag)
            raw_card = rest[:end_pos]
            line = extract_tool_text(raw_card, self.rollout_id)
            if line:
                if output_parts and not output_parts[-1].endswith("\n"):
                    output_parts[-1] += "\n"
                output_parts.append(f"{line}\n")
            rest = rest[end_pos:]

        return "".join(output_parts)

    def _filter_token(self, token: str) -> str:
        """过滤 token 中的特殊标签（如 <grok:render>...</grok:render>），支持跨 token 的标签过滤"""
        if not token:
            return token

        # 先处理工具调用卡片提取（提取而非删除）
        if self.tool_usage_enabled:
            token = self._filter_tool_card(token)
            if not token:
                return ""

        if not self.filter_tags:
            return token

        result = []
        i = 0
        while i < len(token):
            char = token[i]

            if self._in_filter_tag:
                self._tag_buffer += char
                if char == ">":
                    if "/>" in self._tag_buffer:
                        self._in_filter_tag = False
                        self._tag_buffer = ""
                    else:
                        for tag in self.filter_tags:
                            if tag == "xai:tool_usage_card":
                                continue
                            if f"</{tag}>" in self._tag_buffer:
                                self._in_filter_tag = False
                                self._tag_buffer = ""
                                break
                i += 1
                continue

            if char == "<":
                remaining = token[i:]
                tag_started = False
                for tag in self.filter_tags:
                    if tag == "xai:tool_usage_card":
                        continue
                    if remaining.startswith(f"<{tag}"):
                        tag_started = True
                        break
                    if len(remaining) < len(tag) + 1:
                        for j in range(1, len(remaining) + 1):
                            if f"<{tag}".startswith(remaining[:j]):
                                tag_started = True
                                break

                if tag_started:
                    self._in_filter_tag = True
                    self._tag_buffer = char
                    i += 1
                    continue

            result.append(char)
            i += 1

        return "".join(result)

    def _sse(
        self,
        content: str = "",
        role: str = None,
        finish: str = None,
        reasoning_content: str = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """构建 SSE 响应"""
        delta: dict[str, Any] = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        elif content:
            delta["content"] = content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls

        chunk = {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": [
                {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            ],
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"

    def _yield_tool_call_delta_chunks(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[str]:
        chunks: list[str] = []
        for idx, call in enumerate(tool_calls):
            call_id = str(call.get("id", ""))
            fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
            fn_name = str(fn.get("name", ""))
            fn_args = str(fn.get("arguments", ""))

            chunks.append(
                self._sse(
                    tool_calls=[
                        {
                            "index": idx,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": ""},
                        }
                    ]
                )
            )

            if fn_args:
                step = 64
                for i in range(0, len(fn_args), step):
                    part = fn_args[i : i + step]
                    chunks.append(
                        self._sse(
                            tool_calls=[
                                {
                                    "index": idx,
                                    "function": {"arguments": part},
                                }
                            ]
                        )
                    )

        chunks.append(self._sse(finish="tool_calls"))
        chunks.append("data: [DONE]\n\n")
        return chunks

    @staticmethod
    def _should_track_answer_chunk(content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        return (
            re.match(
                r"^(?:\[[^\]]+\])?\[(WebSearch|SearchImage|AgentThink)\]",
                text,
            )
            is None
        )

    def _track_answer_chunk(self, content: str, in_think: bool) -> None:
        if in_think:
            return
        if self._should_track_answer_chunk(content):
            self.answer_text += content

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _model_response_delta(self, final_text: str) -> str:
        """计算 modelResponse 相对已输出文本的增量，避免尾包聚合重复。"""
        final_clean = str(final_text or "").strip()
        if not final_clean:
            return ""

        emitted = self.answer_text.strip()
        if not emitted:
            return final_clean

        if final_clean == emitted:
            return ""
        if final_clean.startswith(emitted):
            return final_clean[len(emitted) :]
        if emitted.endswith(final_clean) or final_clean in emitted:
            return ""

        compact_final = self._compact_text(final_clean)
        compact_emitted = self._compact_text(emitted)
        if compact_final and compact_final in compact_emitted:
            return ""

        max_overlap = min(len(emitted), len(final_clean))
        for size in range(max_overlap, 0, -1):
            if emitted[-size:] == final_clean[:size]:
                return final_clean[size:]

        if len(final_clean) <= len(emitted):
            return ""

        prefix_len = 0
        max_len = min(len(final_clean), len(emitted))
        while (
            prefix_len < max_len
            and final_clean[prefix_len] == emitted[prefix_len]
        ):
            prefix_len += 1
        if prefix_len > 0:
            return final_clean[prefix_len:]

        return final_clean

    def _resolve_thinking_state(self, resp: dict[str, Any]) -> bool:
        """统一解析 thinking 状态，兼容 isThinking / reasoningStatus。"""
        if "isThinking" in resp:
            state = bool(resp.get("isThinking"))
            self._last_thinking_state = state
            return state

        status = resp.get("reasoningStatus")
        if isinstance(status, str):
            lowered = status.strip().lower()
            if lowered in {"enabled", "disabled"}:
                state = lowered == "enabled"
                self._last_thinking_state = state
                return state

        # 对 token 事件，若上游未显式带状态，沿用上一段 thinking 状态。
        if resp.get("token") is not None:
            return self._last_thinking_state

        return False

    async def process(
        self, response: AsyncIterable[bytes]
    ) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        idle_timeout = get_config("timeout.stream_idle_timeout")

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_stream_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})
                is_thinking = self._resolve_thinking_state(resp)

                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid
                if rid := resp.get("rolloutId"):
                    self.rollout_id = str(rid)

                token = resp.get("token")
                if token is None:
                    token = ""

                if self._tool_branch_enabled and not self._tool_mode_determined and token:
                    self._tool_gate_buffer += token
                    gate_trimmed = self._tool_gate_buffer.lstrip()

                    # 命中前缀：进入 tool 分支
                    if gate_trimmed.startswith(self.tool_call_prefix):
                        self._tool_mode_enabled = True
                        self._tool_mode_determined = True
                        self._tool_raw_buffer = self._tool_gate_buffer
                        continue

                    # 仍可能是前缀的部分片段：继续等待
                    if self.tool_call_prefix.startswith(gate_trimmed):
                        continue

                    # 明确不是前缀：回落到普通文本，并先冲刷缓冲
                    self._tool_mode_enabled = False
                    self._tool_mode_determined = True
                    if not self.role_sent:
                        yield self._sse(role="assistant")
                        self.role_sent = True
                    buffered = self._filter_token(self._tool_gate_buffer)
                    if buffered:
                        yield self._sse(buffered)
                        self._track_answer_chunk(buffered, in_think=False)
                    self._tool_gate_buffer = ""
                    continue

                if self._tool_branch_enabled and self._tool_mode_enabled:
                    if token:
                        self._tool_raw_buffer += token
                    continue

                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True

                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if not self.show_think:
                        continue
                    self.image_think_active = True
                    if not self.think_opened:
                        yield self._sse("<think>\n")
                        self.think_opened = True
                    idx = img.get("imageIndex", 0) + 1
                    progress = img.get("progress", 0)
                    yield self._sse(
                        f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                    )
                    continue

                # 多专家深度思考
                agent_name = resp.get("rolloutId", "AI")
                if tool_card := resp.get("toolUsageCard"):
                    if self.show_think:
                        if query := (
                            tool_card.get("webSearch", {})
                            .get("args", {})
                            .get("query")
                        ):
                            yield self._sse(
                                reasoning_content=f"{agent_name}: 正在搜索 `{query}` 中...\n"
                            )
                        if agent_talk := (
                            tool_card.get("chatroomSend", {})
                            .get("args", {})
                            .get("message")
                        ):
                            yield self._sse(
                                reasoning_content=f"{agent_name}: {agent_talk}\n"
                            )

                # 搜索结果
                if search_results := resp.get("webSearchResults"):
                    if self.show_think:
                        if results := search_results.get("results", []):
                            if results:
                                yield self._sse(
                                    reasoning_content=f"{agent_name} 搜索结果:\n"
                                )
                            for result in results:
                                url = result.get("url", "")
                                title = result.get("title", "")
                                preview = result.get("preview", "")
                                yield self._sse(
                                    reasoning_content=f"[{title}]({url})\n> {preview}\n\n"
                                )

                # modelResponse
                if model_response := resp.get("modelResponse"):
                    if self.image_think_active:
                        self.image_think_active = False
                    if self.think_opened:
                        yield self._sse("\n</think>\n")
                        self.think_opened = False

                    msg = model_response.get("message")
                    if isinstance(msg, str):
                        delta_text = self._model_response_delta(msg)
                        if delta_text:
                            yield self._sse(delta_text)
                            self._track_answer_chunk(delta_text, in_think=False)

                    # 处理生成的图片
                    for url in _collect_image_urls(model_response):
                        parts = url.split("/")
                        img_id = parts[-2] if len(parts) >= 2 else "image"

                        if self.image_format == "base64":
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.to_base64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    yield self._sse(f"![{img_id}]({base64_data})\n")
                                else:
                                    final_url = await self.process_url(url, "image")
                                    yield self._sse(f"![{img_id}]({final_url})\n")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert image to base64, falling back to URL: {e}"
                                )
                                final_url = await self.process_url(url, "image")
                                yield self._sse(f"![{img_id}]({final_url})\n")
                        else:
                            final_url = await self.process_url(url, "image")
                            yield self._sse(f"![{img_id}]({final_url})\n")

                    if (
                        (meta := model_response.get("metadata", {}))
                        .get("llm_info", {})
                        .get("modelHash")
                    ):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue

                # cardAttachment（搜索结果卡片）
                if card := resp.get("cardAttachment"):
                    json_data = card.get("jsonData")
                    if isinstance(json_data, str) and json_data.strip():
                        try:
                            card_data = orjson.loads(json_data)
                        except orjson.JSONDecodeError:
                            card_data = None
                        if isinstance(card_data, dict):
                            image = card_data.get("image") or {}
                            original = image.get("original")
                            title = image.get("title") or ""
                            if original:
                                title_safe = title.replace("\n", " ").strip()
                                if title_safe:
                                    yield self._sse(f"![{title_safe}]({original})\n")
                                else:
                                    yield self._sse(f"![image]({original})\n")
                    continue

                # 普通 token
                if token:
                    filtered = self._filter_token(token)
                    if not filtered:
                        continue
                    in_think = is_thinking or self.image_think_active
                    if in_think:
                        if not self.show_think:
                            continue
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                    else:
                        if self.think_opened:
                            yield self._sse("\n</think>\n")
                            self.think_opened = False
                    yield self._sse(filtered)
                    self._track_answer_chunk(filtered, in_think=in_think)

            if self._tool_branch_enabled and self._tool_mode_enabled:
                plan = self._tool_planner.plan(
                    self._tool_raw_buffer,
                    tools=self.tools,
                    tool_choice=self.tool_choice,
                )
                if plan.error and self.tool_call_strict:
                    raise UpstreamException(
                        message=f"Invalid tool call payload: {plan.error}",
                        details={"error": plan.error, "type": "invalid_tool_call_payload"},
                    )
                if plan.is_tool_call and plan.tool_calls:
                    if not self.role_sent:
                        yield self._sse(role="assistant")
                        self.role_sent = True
                    chunks = self._yield_tool_call_delta_chunks(
                        [tc.to_openai_dict() for tc in plan.tool_calls]
                    )
                    for item in chunks:
                        yield item
                    return

            if self.think_opened:
                yield self._sse("</think>\n")
                self.think_opened = False
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            logger.debug("Stream cancelled by client", extra={"model": self.model})
        except StreamIdleTimeoutError as e:
            raise UpstreamException(
                message=f"Stream idle timeout after {e.idle_seconds}s",
                status_code=504,
                details={
                    "error": str(e),
                    "type": "stream_idle_timeout",
                    "idle_seconds": e.idle_seconds,
                },
            )
        except RequestsError as e:
            if _is_http2_stream_error(e):
                logger.warning(f"HTTP/2 stream error: {e}", extra={"model": self.model})
                raise UpstreamException(
                    message="Upstream connection closed unexpectedly",
                    status_code=502,
                    details={"error": str(e), "type": "http2_stream_error"},
                )
            logger.error(f"Stream request error: {e}", extra={"model": self.model})
            raise UpstreamException(
                message=f"Upstream request failed: {e}",
                status_code=502,
                details={"error": str(e)},
            )
        except Exception as e:
            logger.error(
                f"Stream processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
            raise
        finally:
            await self.close()


class CollectProcessor(BaseProcessor):
    """非流式响应处理器"""

    def __init__(
        self,
        model: str,
        token: str = "",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format")
        self.filter_tags = get_config("chat.filter_tags")
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.tool_call_simulation_enabled = bool(
            get_config("chat.tool_call_simulation_enabled", True)
        )
        self.tool_call_prefix = get_config("chat.tool_call_prefix", "<|tool_call|>")
        self.tool_call_strict = bool(get_config("chat.tool_call_strict", True))
        self._tool_planner = ToolCallPlanner(
            prefix=self.tool_call_prefix,
            strict=self.tool_call_strict,
        )

    def _filter_content(self, content: str) -> str:
        """过滤内容中的特殊标签，对 xai:tool_usage_card 提取工具调用信息"""
        if not content or not self.filter_tags:
            return content

        result = content

        # 对 xai:tool_usage_card 特殊处理：提取工具信息而非删除
        if "xai:tool_usage_card" in self.filter_tags:
            rollout_id = ""
            rollout_match = re.search(
                r"<rolloutId>(.*?)</rolloutId>", result, flags=re.DOTALL
            )
            if rollout_match:
                rollout_id = rollout_match.group(1).strip()

            result = re.sub(
                r"<xai:tool_usage_card[^>]*>.*?</xai:tool_usage_card>",
                lambda match: (
                    f"{extract_tool_text(match.group(0), rollout_id)}\n"
                    if extract_tool_text(match.group(0), rollout_id)
                    else ""
                ),
                result,
                flags=re.DOTALL,
            )

        # 其他标签直接删除
        for tag in self.filter_tags:
            if tag == "xai:tool_usage_card":
                continue
            pattern = rf"<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>|<{re.escape(tag)}[^>]*/>"
            result = re.sub(pattern, "", result, flags=re.DOTALL)

        return result

    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集完整响应"""
        response_id = ""
        fingerprint = ""
        content = ""
        token_buffer = ""
        idle_timeout = get_config("timeout.stream_idle_timeout")
        reasoning_content = ""

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_stream_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})

                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    response_id = rid
                if (token := resp.get("token")) is not None and token:
                    token_buffer += token

                agent_name = resp.get("rolloutId", "AI")
                if tool_card := resp.get("toolUsageCard"):
                    if query := (
                        tool_card.get("webSearch", {})
                        .get("args", {})
                        .get("query")
                    ):
                        reasoning_content += f"{agent_name}: 正在搜索 `{query}` 中...\n"
                    if agent_talk := (
                        tool_card.get("chatroomSend", {})
                        .get("args", {})
                        .get("message")
                    ):
                        reasoning_content += f"{agent_name}: {agent_talk}\n"

                if search_results := resp.get("webSearchResults"):
                    if results := search_results.get("results", []):
                        if results:
                            reasoning_content += f"{agent_name} 搜索结果:\n"
                        for result in results:
                            url = result.get("url", "")
                            title = result.get("title", "")
                            preview = result.get("preview", "")
                            reasoning_content += f"[{title}]({url})\n> {preview}\n\n"

                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "") or response_id
                    if isinstance(mr.get("message"), str):
                        content = mr.get("message", "")

                    if model_search := mr.get("webSearchResults"):
                        model_results = []
                        if isinstance(model_search, dict):
                            model_results = model_search.get("results", []) or []
                        elif isinstance(model_search, list):
                            model_results = model_search
                        if model_results:
                            reasoning_content += "最终使用的搜索结果:\n"
                            for result in model_results:
                                url = result.get("url", "")
                                title = result.get("title", "")
                                preview = result.get("preview", "")
                                reasoning_content += f"[{title}]({url})\n> {preview}\n\n"

                    if urls := _collect_image_urls(mr):
                        content += "\n"
                        for url in urls:
                            parts = url.split("/")
                            img_id = parts[-2] if len(parts) >= 2 else "image"

                            if self.image_format == "base64":
                                try:
                                    dl_service = self._get_dl()
                                    base64_data = await dl_service.to_base64(
                                        url, self.token, "image"
                                    )
                                    if base64_data:
                                        content += f"![{img_id}]({base64_data})\n"
                                    else:
                                        final_url = await self.process_url(url, "image")
                                        content += f"![{img_id}]({final_url})\n"
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to convert image to base64, falling back to URL: {e}"
                                    )
                                    final_url = await self.process_url(url, "image")
                                    content += f"![{img_id}]({final_url})\n"
                            else:
                                final_url = await self.process_url(url, "image")
                                content += f"![{img_id}]({final_url})\n"

                    if (
                        (meta := mr.get("metadata", {}))
                        .get("llm_info", {})
                        .get("modelHash")
                    ):
                        fingerprint = meta["llm_info"]["modelHash"]

        except asyncio.CancelledError:
            logger.debug("Collect cancelled by client", extra={"model": self.model})
        except StreamIdleTimeoutError as e:
            logger.warning(f"Collect idle timeout: {e}", extra={"model": self.model})
        except RequestsError as e:
            if _is_http2_stream_error(e):
                logger.warning(
                    f"HTTP/2 stream error in collect: {e}", extra={"model": self.model}
                )
            else:
                logger.error(f"Collect request error: {e}", extra={"model": self.model})
        except Exception as e:
            logger.error(
                f"Collect processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
        finally:
            await self.close()

        tool_calls = None
        raw_tool_text = ""
        candidates: list[str] = []
        if isinstance(token_buffer, str) and token_buffer.strip():
            candidates.append(token_buffer)
        if isinstance(content, str) and content.strip():
            candidates.append(content)

        for candidate in candidates:
            if candidate.lstrip().startswith(self.tool_call_prefix):
                raw_tool_text = candidate
                break

        if (
            self.tool_call_simulation_enabled
            and self.tools
            and self.tool_choice != "none"
            and raw_tool_text
        ):
            plan = self._tool_planner.plan(
                raw_tool_text,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )
            if plan.error and self.tool_call_strict:
                raise UpstreamException(
                    message=f"Invalid tool call payload: {plan.error}",
                    details={"error": plan.error, "type": "invalid_tool_call_payload"},
                )
            if plan.is_tool_call and plan.tool_calls:
                tool_calls = [tc.to_openai_dict() for tc in plan.tool_calls]

        content = self._filter_content(content)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "refusal": None,
            "annotations": [],
        }
        finish_reason = "stop"
        if tool_calls:
            message["content"] = ""
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        if reasoning_content and get_config("chat.thinking"):
            message["reasoning_content"] = reasoning_content

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                },
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        }


__all__ = ["StreamProcessor", "CollectProcessor"]
