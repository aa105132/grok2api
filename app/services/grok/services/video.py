"""
Grok Video WebSocket service.
"""

import asyncio
import json
import re
import ssl
import time
import uuid
from typing import AsyncGenerator, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from aiohttp_socks import ProxyConnector

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.utils.headers import build_sso_cookie

WS_URL = "wss://grok.com/ws/video/listen"


class _BlockedError(Exception):
    pass


class VideoWSService:
    """Grok Video WebSocket service."""

    def __init__(self):
        self._ssl_context = ssl.create_default_context()
        self._url_pattern = re.compile(r"/videos?/([a-f0-9-]+)\.(mp4|webm)")

    def _resolve_proxy(self) -> tuple[aiohttp.BaseConnector, Optional[str]]:
        proxy_url = get_config("network.base_proxy_url")
        if not proxy_url:
            return aiohttp.TCPConnector(ssl=self._ssl_context), None

        scheme = urlparse(proxy_url).scheme.lower()
        if scheme.startswith("socks"):
            logger.info(f"Using SOCKS proxy for video WS: {proxy_url}")
            return ProxyConnector.from_url(proxy_url, ssl=self._ssl_context), None

        logger.info(f"Using HTTP proxy for video WS: {proxy_url}")
        return aiohttp.TCPConnector(ssl=self._ssl_context), proxy_url

    def _get_ws_headers(self, token: str) -> Dict[str, str]:
        cookie = build_sso_cookie(token, include_rw=True)
        user_agent = get_config("security.user_agent")
        return {
            "Cookie": cookie,
            "Origin": "https://grok.com",
            "User-Agent": user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _extract_video_id(self, url: str) -> Optional[str]:
        match = self._url_pattern.search(url or "")
        return match.group(1) if match else None

    async def stream(
        self,
        token: str,
        prompt: str,
        post_id: str,
        aspect_ratio: str = "3:2",
        video_length: int = 6,
        resolution: str = "480p",
        preset: str = "normal",
        max_retries: int = None,
    ) -> AsyncGenerator[Dict[str, object], None]:
        retries = max(1, max_retries if max_retries is not None else 1)
        logger.info(
            f"Video WS generation: prompt='{prompt[:50]}...', ratio={aspect_ratio}, length={video_length}s"
        )

        for attempt in range(retries):
            try:
                yielded_any = False
                async for item in self._stream_once(
                    token, prompt, post_id, aspect_ratio, video_length, resolution, preset
                ):
                    yielded_any = True
                    yield item
                return
            except _BlockedError:
                if yielded_any or attempt + 1 >= retries:
                    if not yielded_any:
                        yield {
                            "type": "error",
                            "error_code": "blocked",
                            "error": "blocked_no_video",
                        }
                    return
                logger.warning(f"Video WebSocket blocked, retry {attempt + 1}/{retries}")
            except Exception as e:
                logger.error(f"Video WebSocket stream failed: {e}")
                return

    async def _stream_once(
        self,
        token: str,
        prompt: str,
        post_id: str,
        aspect_ratio: str,
        video_length: int,
        resolution: str,
        preset: str,
    ) -> AsyncGenerator[Dict[str, object], None]:
        request_id = str(uuid.uuid4())
        headers = self._get_ws_headers(token)
        timeout = float(get_config("network.timeout"))
        blocked_seconds = float(get_config("video.video_ws_blocked_seconds"))

        mode_map = {
            "fun": "--mode=extremely-crazy",
            "normal": "--mode=normal",
            "spicy": "--mode=extremely-spicy-or-crazy",
        }
        mode_flag = mode_map.get(preset, "--mode=custom")

        try:
            connector, proxy = self._resolve_proxy()
        except Exception as e:
            logger.error(f"Video WebSocket proxy setup failed: {e}")
            return

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.ws_connect(
                    WS_URL,
                    headers=headers,
                    heartbeat=20,
                    receive_timeout=timeout,
                    proxy=proxy,
                ) as ws:
                    message = {
                        "type": "conversation.item.create",
                        "timestamp": int(time.time() * 1000),
                        "item": {
                            "type": "message",
                            "content": [
                                {
                                    "requestId": request_id,
                                    "text": f"{prompt} {mode_flag}",
                                    "type": "input_text",
                                    "properties": {
                                        "parentPostId": post_id,
                                        "aspectRatio": aspect_ratio,
                                        "videoLength": video_length,
                                        "resolutionName": resolution,
                                    },
                                }
                            ],
                        },
                    }

                    await ws.send_json(message)
                    logger.info(f"Video WebSocket request sent: post_id={post_id}")

                    completed = False
                    start_time = last_activity = time.time()
                    progress_received_time = None

                    while time.time() - start_time < timeout:
                        try:
                            ws_msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                        except asyncio.TimeoutError:
                            if (
                                progress_received_time
                                and not completed
                                and time.time() - progress_received_time
                                > min(30, blocked_seconds)
                            ):
                                raise _BlockedError()
                            if completed and time.time() - last_activity > 10:
                                logger.info("Video WebSocket idle timeout, video completed")
                                break
                            continue

                        if ws_msg.type == aiohttp.WSMsgType.TEXT:
                            last_activity = time.time()
                            msg = json.loads(ws_msg.data)
                            msg_type = msg.get("type")

                            if msg_type == "video_progress":
                                progress = msg.get("progress", 0)
                                if progress_received_time is None:
                                    progress_received_time = time.time()
                                
                                yield {
                                    "type": "video_progress",
                                    "progress": progress,
                                }
                                
                                logger.debug(f"Video progress: {progress}%")

                            elif msg_type == "video":
                                video_url = msg.get("url", "")
                                thumbnail_url = msg.get("thumbnailUrl", "")
                                video_id = self._extract_video_id(video_url) or uuid.uuid4().hex
                                
                                completed = True
                                logger.info(f"Video completed: {video_url[:80]}")
                                
                                yield {
                                    "type": "video",
                                    "video_id": video_id,
                                    "video_url": video_url,
                                    "thumbnail_url": thumbnail_url,
                                    "is_final": True,
                                }
                                break

                            elif msg_type == "error":
                                logger.warning(
                                    f"Video WebSocket error: {msg.get('err_code', '')} - {msg.get('err_msg', '')}"
                                )
                                yield {
                                    "type": "error",
                                    "error_code": msg.get("err_code", ""),
                                    "error": msg.get("err_msg", ""),
                                }
                                return

                            if (
                                progress_received_time
                                and not completed
                                and time.time() - progress_received_time > blocked_seconds
                            ):
                                raise _BlockedError()

                        elif ws_msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning(f"Video WebSocket closed/error: {ws_msg.type}")
                            yield {
                                "type": "error",
                                "error_code": "ws_closed",
                                "error": f"websocket closed: {ws_msg.type}",
                            }
                            break

        except aiohttp.ClientError as e:
            logger.error(f"Video WebSocket connection error: {e}")
            yield {"type": "error", "error_code": "connection_failed", "error": str(e)}


video_ws_service = VideoWSService()

__all__ = ["video_ws_service", "VideoWSService"]