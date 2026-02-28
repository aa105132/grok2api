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
        self.think_opened: bool = False
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

        if think is None:
            self.show_think = get_config("chat.thinking")
        else:
            self.show_think = think

    def _filter_token(self, token: str) -> str:
        """过滤 token 中的特殊标签（如 <grok:render>...</grok:render>），支持跨 token 的标签过滤"""
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
        return f"data: {orjson.dumps(chunk).decode("utf-8")}\n\n"

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

                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid

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
                if img := resp.get("streamingImageGenerationResponse", None):
                    if self.show_think:
                        if not self.think_opened:
                            self.think_opened = True
                        idx = img.get("imageIndex", 0) + 1
                        progress = img.get("progress", 0)
                        yield self._sse(
                            reasoning_content=(
                                f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                            )
                        )
                    continue
                
                # 多专家深度思考
                agent_name = resp.get("rolloutId", "AI")
                if tool_card := resp.get("toolUsageCard", None):
                    if self.show_think:
                        # 搜索内容
                        if query := tool_card.get("webSearch", {}).get("args", {}).get("query", None):
                            yield self._sse(reasoning_content=f"{agent_name}: 正在搜索 `{query}` 中...\n")
                        
                        # 专家对话
                        if agent_talk := tool_card.get("chatroomSend", {}).get("args", {}).get("message", None):
                            yield self._sse(reasoning_content=f"{agent_name}: {agent_talk}\n")
                
                # 搜索结果
                if search_results := resp.get("webSearchResults", None):
                    if self.show_think:
                        if results := search_results.get("results", []):
                            if results:
                                yield self._sse(reasoning_content=f"{agent_name} 搜索结果: \n")
                            
                            for result in results:
                                url = result.get("url", "")
                                title = result.get("title", "")
                                preview = result.get("preview", "")
                                yield self._sse(reasoning_content=f"[{title}]({url})\n> {preview}\n\n")
                
                # modelResponse
                if model_response := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        if msg := model_response.get("message"):
                            yield self._sse(reasoning_content=msg + "\n")
                        
                        # 处理搜索结果
                        if search_results := model_response.get("webSearchResults", None):
                            if self.show_think:
                                if results:
                                    yield self._sse(reasoning_content="最终使用的搜索结果: \n")
                                
                                for result in results:
                                    url = result.get("url", "")
                                    title = result.get("title", "")
                                    preview = result.get("preview", "")
                                    yield self._sse(reasoning_content=f"[{title}]({url})\n> {preview}\n\n")
                        
                        self.think_opened = False

                    

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

                # 普通 token
                if token:
                    filtered = self._filter_token(token)
                    if filtered:
                        yield self._sse(filtered)

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
                exc_info=True,
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
        """过滤内容中的特殊标签"""
        if not content or not self.filter_tags:
            return content

        result = content
        for tag in self.filter_tags:
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
                
                # 多专家深度思考
                agent_name = resp.get("rolloutId", "AI")
                if tool_card := resp.get("toolUsageCard", None):
                    # 搜索内容
                    if query := tool_card.get("webSearch", {}).get("args", {}).get("query", None):
                        reasoning_content += f"{agent_name}：正在搜索 `{query}` 中...\n"
                    
                    # 专家对话
                    if agent_talk := tool_card.get("chatroomSend", {}).get("args", {}).get("message", None):
                        reasoning_content += f"{agent_name}: {agent_talk}\n"
                
                # 搜索结果
                if search_results := resp.get("webSearchResults", None):
                    if results := search_results.get("results", []):
                        if results:
                            reasoning_content += f"{agent_name} 搜索结果: \n"
                        
                        for result in results:
                            url = result.get("url", "")
                            title = result.get("title", "")
                            preview = result.get("preview", "")
                            reasoning_content += f"[{title}]({url})\n> {preview}\n\n"

                if model_response := resp.get("modelResponse"):
                    response_id = model_response.get("responseId", "")
                    content = model_response.get("message", "")

                    if search_results := model_response.get("webSearchResults", None):
                        reasoning_content += "最终使用的搜索结果: \n"
                        
                        for result in search_results:
                            url = result.get("url", "")
                            title = result.get("title", "")
                            preview = result.get("preview", "")
                            reasoning_content += f"[{title}]({url})\n> {preview}\n\n"

                    if urls := _collect_image_urls(model_response):
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
                        (meta := model_response.get("metadata", {}))
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
                exc_info=True,
            )
        finally:
            await self.close()

        tool_calls = None
        raw_tool_text = ""

        # 优先选择命中前缀的候选文本，避免 token_buffer 存在噪声导致误判
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
