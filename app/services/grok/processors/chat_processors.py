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

    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: str = None
        self.fingerprint: str = ""
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.normal_token_emitted: bool = False
        self.filter_tags = get_config("chat.filter_tags")
        self.image_format = get_config("app.image_format")
        self._tag_buffer: str = ""
        self._in_filter_tag: bool = False

        if think is None:
            self.show_think = get_config("chat.thinking")
        else:
            self.show_think = think

    @staticmethod
    def _parse_bool_flag(value: Any) -> bool | None:
        """将多种布尔表示归一化为 bool。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return None

    def _extract_reasoning_flag(self, payload: Any) -> bool | None:
        """
        尝试从上游响应中提取“当前 token 是否为思维链”标记。

        兼容不同字段命名，无法识别时返回 None。
        """
        if not isinstance(payload, dict):
            return None

        bool_keys = (
            "isThinking",
            "thinking",
            "inThinking",
            "isReasoning",
            "reasoning",
            "inReasoning",
        )
        for key in bool_keys:
            parsed = self._parse_bool_flag(payload.get(key))
            if parsed is not None:
                return parsed

        type_keys = ("tokenType", "streamType", "responseType", "phase", "state", "type")
        for key in type_keys:
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower().replace("-", "_")
            if any(mark in normalized for mark in ("think", "reason", "cot", "chain_of_thought")):
                return True
            if any(mark in normalized for mark in ("final", "answer", "output")):
                return False

        nested_keys = ("metadata", "tokenInfo", "streamingMetadata", "llmInfo")
        for key in nested_keys:
            nested = payload.get(key)
            parsed = self._extract_reasoning_flag(nested)
            if parsed is not None:
                return parsed

        return None

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

    def _sse(self, content: str = "", role: str = None, finish: str = None) -> str:
        """构建 SSE 响应"""
        delta = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        elif content:
            delta["content"] = content

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

                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True

                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        idx = img.get("imageIndex", 0) + 1
                        progress = img.get("progress", 0)
                        yield self._sse(
                            f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                        )
                    continue

                # modelResponse
                if mr := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        yield self._sse("</think>\n")
                        self.think_opened = False

                    # 某些上游实现只在 modelResponse.message 给出最终答案；
                    # 当之前没有输出过普通 token 时，用该字段兜底。
                    if not self.normal_token_emitted and (msg := mr.get("message")):
                        filtered_msg = self._filter_token(str(msg))
                        if filtered_msg:
                            yield self._sse(filtered_msg)
                            self.normal_token_emitted = True

                    # 处理生成的图片
                    for url in _collect_image_urls(mr):
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
                        (meta := mr.get("metadata", {}))
                        .get("llm_info", {})
                        .get("modelHash")
                    ):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue

                # 普通 token
                if (token := resp.get("token")) is not None:
                    if token:
                        filtered = self._filter_token(token)
                        if filtered:
                            reasoning_flag = self._extract_reasoning_flag(resp)

                            if reasoning_flag is True:
                                if self.show_think:
                                    if not self.think_opened:
                                        yield self._sse("<think>\n")
                                        self.think_opened = True
                                    yield self._sse(filtered)
                                continue

                            if self.think_opened and self.show_think:
                                yield self._sse("</think>\n")
                                self.think_opened = False

                            self.normal_token_emitted = True
                            yield self._sse(filtered)

            if self.think_opened:
                yield self._sse("</think>\n")
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

    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format")
        self.filter_tags = get_config("chat.filter_tags")

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

                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")

                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "")
                    content = mr.get("message", "")

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

        content = self._filter_content(content)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                        "annotations": [],
                    },
                    "finish_reason": "stop",
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
