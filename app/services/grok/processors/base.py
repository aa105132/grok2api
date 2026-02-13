"""
Base processor utilities for stream parsing and asset URL handling.
"""

import asyncio
import base64
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from typing import Any, AsyncGenerator, AsyncIterable, List, Optional, TypeVar

from app.core.config import get_config
from app.core.logger import logger
from app.core.storage import DATA_DIR
from app.services.grok.services.assets import DownloadService

ASSET_URL = "https://assets.grok.com/"
T = TypeVar("T")

_CACHE_BASE_DIR = DATA_DIR / "tmp"
_IMAGE_CACHE_DIR = _CACHE_BASE_DIR / "image"
_VIDEO_CACHE_DIR = _CACHE_BASE_DIR / "video"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
}


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

    @staticmethod
    def _cache_dir(media_type: str) -> Path:
        return _VIDEO_CACHE_DIR if media_type == "video" else _IMAGE_CACHE_DIR

    @staticmethod
    def _allowed_exts(media_type: str) -> set[str]:
        return _VIDEO_EXTS if media_type == "video" else _IMAGE_EXTS

    @staticmethod
    def _default_ext(media_type: str) -> str:
        return ".mp4" if media_type == "video" else ".jpg"

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path or "").strip()
        if normalized.startswith("http"):
            parsed = urlparse(normalized)
            normalized = parsed.path or "/"
            if parsed.query:
                normalized = f"{normalized}?{parsed.query}"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    @staticmethod
    def _path_suffix(path: str, media_type: str) -> str:
        raw_path = path.split("?", 1)[0]
        ext = Path(raw_path).suffix.lower()
        if ext in BaseProcessor._allowed_exts(media_type):
            return ext
        return ""

    @staticmethod
    def _parse_data_uri(data_uri: str) -> tuple[str, str]:
        if "," not in data_uri:
            return "application/octet-stream", ""
        header, b64 = data_uri.split(",", 1)
        mime = "application/octet-stream"
        if header.startswith("data:"):
            mime = header[5:].split(";", 1)[0] or mime
        return mime, b64

    @staticmethod
    async def _write_file(file_path: Path, content: bytes) -> None:
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(file_path.write_bytes, content)

    def _public_file_url(self, media_type: str, filename: str) -> str:
        relative = f"/v1/files/{media_type}/{quote(filename)}"
        app_url = str(self.app_url or "").strip().rstrip("/")
        if app_url:
            return f"{app_url}{relative}"
        return relative

    async def process_url(self, path: str, media_type: str = "image") -> str:
        """Download and cache asset locally, returning a /v1/files/* URL."""
        import asyncio

        normalized_path = self._normalize_path(path)
        cache_dir = self._cache_dir(media_type)
        key = hashlib.sha1(
            f"{media_type}:{normalized_path}".encode("utf-8")
        ).hexdigest()

        ext_from_path = self._path_suffix(normalized_path, media_type)
        if ext_from_path:
            cached = cache_dir / f"{key}{ext_from_path}"
            if cached.exists():
                return self._public_file_url(media_type, cached.name)
        else:
            existing = next(cache_dir.glob(f"{key}.*"), None)
            if existing and existing.is_file():
                return self._public_file_url(media_type, existing.name)

        # 重试逻辑
        max_retries = 5
        retry_delays = [2, 3, 5, 8, 10]  # 增加重试次数和延迟(总共28秒)

        for attempt in range(max_retries):
            try:
                dl_service = self._get_dl()
                data_uri = await dl_service.to_base64(normalized_path, self.token, media_type)
                mime, b64 = self._parse_data_uri(data_uri)
                if not b64:
                    raise ValueError("empty base64 payload")

                ext = ext_from_path or _MIME_TO_EXT.get(mime, self._default_ext(media_type))
                filename = f"{key}{ext}"
                file_path = cache_dir / filename

                if not file_path.exists():
                    binary = base64.b64decode(b64)
                    await self._write_file(file_path, binary)

                return self._public_file_url(media_type, filename)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.info(
                        f"Cache asset attempt {attempt + 1}/{max_retries} failed for {normalized_path}, "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Cache asset failed after {max_retries} attempts, fallback to upstream URL: {e}"
                    )
                    # imagine-public 资源不在 assets.grok.com，fallback 时保留正确主机。
                    if normalized_path.startswith("/imagine-public/"):
                        return f"https://grok.com{normalized_path}"
                    return f"{ASSET_URL.rstrip('/')}{normalized_path}"


__all__ = [
    "BaseProcessor",
    "StreamIdleTimeoutError",
    "_with_idle_timeout",
    "_normalize_stream_line",
    "_collect_image_urls",
    "_is_http2_stream_error",
]
