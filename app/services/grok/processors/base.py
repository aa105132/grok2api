"""
Base processor utilities for stream parsing and asset URL handling.
"""

import asyncio
import re
import time
from typing import Any, AsyncGenerator, AsyncIterable, List, Optional, TypeVar

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.services.assets import DownloadService

ASSET_URL = "https://assets.grok.com/"
T = TypeVar("T")


def _is_http2_stream_error(e: Exception) -> bool:
    """Return True when an exception looks like an HTTP/2 stream reset/close."""
    err_str = str(e).lower()
    return "http/2" in err_str or "curl: (92)" in err_str or "stream" in err_str


def _normalize_stream_line(line: Any) -> Optional[str]:
    """Normalize stream line content and strip SSE prefixes/no-op lines."""
    if line is None:
        return None
    if isinstance(line, (bytes, bytearray)):
        text = line.decode("utf-8", errors="ignore")
    else:
        text = str(line)
    text = text.strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if text == "[DONE]":
        return None
    return text


def _collect_image_urls(obj: Any) -> List[str]:
    """Recursively collect image URLs from upstream responses."""
    urls: List[str] = []
    seen = set()

    def add(url: str):
        if not url or url in seen:
            return
        seen.add(url)
        urls.append(url)

    def maybe_add_image_url(candidate: str, key_hint: str = ""):
        if not isinstance(candidate, str):
            return
        s = candidate.strip()
        if not s:
            return

        key_hint = (key_hint or "").lower()
        # Avoid echoing request-side image references as generated outputs.
        if "reference" in key_hint:
            return

        lower = s.lower().split("?", 1)[0]
        has_image_ext = lower.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        )
        likely_asset = (
            "assets.grok.com" in lower
            or lower.startswith("users/")
            or lower.startswith("/users/")
            or "/images/" in lower
            or "/generated/" in lower
        )
        is_content_path = lower.endswith("/content")

        if likely_asset and (has_image_ext or is_content_path):
            add(s)
            return

        # Some fields may embed URL text instead of a plain URL field.
        for m in re.findall(
            r"https?://assets\.grok\.com/[^\s\"'<>]+", s, flags=re.IGNORECASE
        ):
            ml = m.lower().split("?", 1)[0]
            if ml.endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", "/content")
            ):
                add(m)

    def walk(value: Any, key_hint: str = ""):
        if isinstance(value, dict):
            for key, item in value.items():
                key_l = str(key).lower()
                if key_l in {
                    "generatedimageurls",
                    "generatedimageurl",
                    "imageurls",
                    "imageurl",
                    "image_url",
                    "image_urls",
                    "outputimageurls",
                    "outputimageurl",
                    "finalimageurls",
                    "finalimageurl",
                    "editedimageurls",
                    "editedimageurl",
                }:
                    if isinstance(item, list):
                        for url in item:
                            maybe_add_image_url(url, key_l)
                    elif isinstance(item, str):
                        maybe_add_image_url(item, key_l)
                    continue
                walk(item, key_l)
        elif isinstance(value, list):
            for item in value:
                walk(item, key_hint)
        elif isinstance(value, str):
            maybe_add_image_url(value, key_hint)

    walk(obj, "")
    return urls


class StreamIdleTimeoutError(Exception):
    """Raised when a stream yields no data within the configured idle timeout."""

    def __init__(self, idle_seconds: float):
        self.idle_seconds = idle_seconds
        super().__init__(f"Stream idle timeout after {idle_seconds}s")


async def _with_idle_timeout(
    iterable: AsyncIterable[T], idle_timeout: float, model: str = ""
) -> AsyncGenerator[T, None]:
    """Wrap an async iterator and enforce an idle timeout per next() call."""
    if idle_timeout <= 0:
        async for item in iterable:
            yield item
        return

    iterator = iterable.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=idle_timeout)
            yield item
        except asyncio.TimeoutError:
            logger.warning(
                f"Stream idle timeout after {idle_timeout}s",
                extra={"model": model, "idle_timeout": idle_timeout},
            )
            raise StreamIdleTimeoutError(idle_timeout)
        except StopAsyncIteration:
            break


class BaseProcessor:
    """Base class for response processors."""

    def __init__(self, model: str, token: str = ""):
        self.model = model
        self.token = token
        self.created = int(time.time())
        self.app_url = get_config("app.app_url")
        self._dl_service: Optional[DownloadService] = None

    def _get_dl(self) -> DownloadService:
        """Get or create a reusable download service."""
        if self._dl_service is None:
            self._dl_service = DownloadService()
        return self._dl_service

    async def close(self):
        """Release download service resources."""
        if self._dl_service:
            await self._dl_service.close()
            self._dl_service = None

    async def process_url(self, path: str, media_type: str = "image") -> str:
        """Normalize to Grok asset URL and return it directly."""
        if path.startswith("http"):
            from urllib.parse import urlparse

            path = urlparse(path).path

        if not path.startswith("/"):
            path = f"/{path}"

        return f"{ASSET_URL.rstrip('/')}{path}"


__all__ = [
    "BaseProcessor",
    "StreamIdleTimeoutError",
    "_with_idle_timeout",
    "_normalize_stream_line",
    "_collect_image_urls",
    "_is_http2_stream_error",
]
