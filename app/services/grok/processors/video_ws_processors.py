"""
视频生成响应处理器（WebSocket）
"""

import time
import uuid
from typing import AsyncGenerator, AsyncIterable, Dict, Optional

import orjson

from app.core.config import get_config
from app.core.logger import logger
from app.core.exceptions import UpstreamException
from .base import BaseProcessor


class VideoWSBaseProcessor(BaseProcessor):
    """WebSocket 视频处理基类"""

    def __init__(self, model: str, token: str = "", show_think: bool = None):
        super().__init__(model, token)
        self.video_format = str(get_config("app.video_format")).lower()
        if show_think is None:
            self.show_think = get_config("chat.thinking")
        else:
            self.show_think = show_think

    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        """构建视频 HTML 标签"""
        import html

        safe_video_url = html.escape(video_url)
        safe_thumbnail_url = html.escape(thumbnail_url)
        poster_attr = f' poster="{safe_thumbnail_url}"' if safe_thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{safe_video_url}" type="video/mp4">
</video>'''


class VideoWSStreamProcessor(VideoWSBaseProcessor):
    """WebSocket 视频流式响应处理器"""

    def __init__(
        self,
        model: str,
        token: str = "",
        show_think: bool = None,
    ):
        super().__init__(model, token, show_think)
        self.response_id: Optional[str] = None
        self.think_opened: bool = False
        self.role_sent: bool = False

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
            "choices": [
                {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            ],
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"

    async def process(
        self, response: AsyncIterable[dict]
    ) -> AsyncGenerator[str, None]:
        """处理视频流式响应"""
        try:
            async for item in response:
                if item.get("type") == "error":
                    message = item.get("error") or "Upstream error"
                    code = item.get("error_code") or "upstream_error"
                    
                    if not self.role_sent:
                        yield self._sse(role="assistant")
                        self.role_sent = True
                    
                    if self.think_opened:
                        yield self._sse("</think>\n")
                        self.think_opened = False
                    
                    yield self._sse(f"Error: {message} (code: {code})")
                    yield self._sse(finish="stop")
                    yield "data: [DONE]\n\n"
                    return

                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True

                if item.get("type") == "video_progress":
                    progress = item.get("progress", 0)
                    
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        yield self._sse(f"正在生成视频中，当前进度{progress}%\n")

                elif item.get("type") == "video":
                    video_url = item.get("video_url", "")
                    thumbnail_url = item.get("thumbnail_url", "")

                    if self.think_opened and self.show_think:
                        yield self._sse("</think>\n")
                        self.think_opened = False

                    if video_url:
                        final_video_url = await self.process_url(video_url, "video")
                        final_thumbnail_url = ""
                        if thumbnail_url:
                            final_thumbnail_url = await self.process_url(
                                thumbnail_url, "image"
                            )

                        if self.video_format == "url":
                            yield self._sse(final_video_url)
                        else:
                            video_html = self._build_video_html(
                                final_video_url, final_thumbnail_url
                            )
                            yield self._sse(video_html)

                        logger.info(f"Video WS generated: {video_url}")

            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(
                f"Video WS stream processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
        finally:
            await self.close()


class VideoWSCollectProcessor(VideoWSBaseProcessor):
    """WebSocket 视频非流式响应处理器"""

    async def process(self, response: AsyncIterable[dict]) -> dict:
        """处理并收集视频响应"""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        content = ""

        try:
            async for item in response:
                if item.get("type") == "error":
                    message = item.get("error") or "Upstream error"
                    raise UpstreamException(message, details=item)

                if item.get("type") == "video":
                    video_url = item.get("video_url", "")
                    thumbnail_url = item.get("thumbnail_url", "")

                    if video_url:
                        final_video_url = await self.process_url(video_url, "video")
                        final_thumbnail_url = ""
                        if thumbnail_url:
                            final_thumbnail_url = await self.process_url(
                                thumbnail_url, "image"
                            )

                        if self.video_format == "url":
                            content = final_video_url
                        else:
                            content = self._build_video_html(
                                final_video_url, final_thumbnail_url
                            )
                        logger.info(f"Video WS generated: {video_url}")
                        break

        except UpstreamException:
            raise
        except Exception as e:
            logger.error(
                f"Video WS collect processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
        finally:
            await self.close()

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


__all__ = ["VideoWSStreamProcessor", "VideoWSCollectProcessor"]