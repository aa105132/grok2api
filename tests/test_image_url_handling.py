import asyncio

from app.core.config import config
from app.services.grok.processors.image_ws_processors import ImageWSCollectProcessor
from app.services.grok.processors.base import _collect_image_urls
from app.services.grok.services.assets import DownloadService


class _DummyResponse:
    status_code = 200
    content = b"abc"
    headers = {"content-type": "image/jpeg"}


class _DummySession:
    def __init__(self):
        self.last_url = None

    async def get(self, url, **kwargs):
        self.last_url = url
        return _DummyResponse()


class _StatusResponse:
    def __init__(self, status_code=200, content=b"abc", content_type="image/jpeg"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


class _QueueSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    async def get(self, url, **kwargs):
        self.urls.append(url)
        if self.responses:
            return self.responses.pop(0)
        return _StatusResponse(status_code=404, content=b"", content_type="text/plain")


class _DummyDownloadService:
    def __init__(self):
        self.calls = []

    async def to_base64(self, file_path, token, media_type="image"):
        self.calls.append(file_path)
        return "data:image/jpeg;base64,AAA"


def test_collect_image_urls_supports_content_paths():
    payload = {
        "generatedImageUrls": [
            "users/u1/generated/a1/content",
            "https://assets.grok.com/users/u1/generated/a2/content",
        ]
    }

    urls = _collect_image_urls(payload)

    assert "users/u1/generated/a1/content" in urls
    assert "https://assets.grok.com/users/u1/generated/a2/content" in urls


def test_collect_image_urls_ignores_reference_urls():
    ref_url = "https://assets.grok.com/users/u1/a3/content"
    generated_url = "users/u1/generated/a4/image.jpg"
    payload = {
        "imageReferences": [ref_url],
        "modelResponse": {"generatedImageUrls": [generated_url]},
    }

    urls = _collect_image_urls(payload)

    assert generated_url in urls
    assert ref_url not in urls


def test_to_base64_accepts_absolute_assets_url():
    dummy = _DummySession()
    svc = DownloadService()

    async def _fake_get_session():
        return dummy

    async def _run():
        await config.load()
        svc._get_session = _fake_get_session  # type: ignore[method-assign]
        svc._build_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        return await svc.to_base64(
            "https://assets.grok.com/users/u1/generated/a5/image.jpg?x=1", "token"
        )

    result = asyncio.run(_run())

    assert dummy.last_url == "https://assets.grok.com/users/u1/generated/a5/image.jpg?x=1"
    assert result.startswith("data:image/jpeg;base64,")


def test_to_base64_imagine_public_prefers_imagine_public_domain():
    dummy = _QueueSession([_StatusResponse(status_code=200)])
    svc = DownloadService()

    async def _fake_get_session():
        return dummy

    async def _run():
        await config.load()
        svc._get_session = _fake_get_session  # type: ignore[method-assign]
        svc._build_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        return await svc.to_base64("/imagine-public/images/demo.jpg", "token")

    result = asyncio.run(_run())

    assert dummy.urls == ["https://imagine-public.x.ai/imagine-public/images/demo.jpg"]
    assert result.startswith("data:image/jpeg;base64,")


def test_to_base64_imagine_public_fallback_to_assets_domain():
    dummy = _QueueSession(
        [
            _StatusResponse(status_code=404, content=b"", content_type="text/plain"),
            _StatusResponse(status_code=404, content=b"", content_type="text/plain"),
            _StatusResponse(status_code=200, content=b"abc", content_type="image/jpeg"),
        ]
    )
    svc = DownloadService()

    async def _fake_get_session():
        return dummy

    async def _run():
        await config.load()
        svc._get_session = _fake_get_session  # type: ignore[method-assign]
        svc._build_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        return await svc.to_base64("/imagine-public/images/demo.jpg", "token")

    result = asyncio.run(_run())

    assert dummy.urls == [
        "https://imagine-public.x.ai/imagine-public/images/demo.jpg",
        "https://grok.com/imagine-public/images/demo.jpg",
        "https://assets.grok.com/imagine-public/images/demo.jpg",
    ]
    assert result.startswith("data:image/jpeg;base64,")


def test_to_base64_skip_non_image_200_response():
    dummy = _QueueSession(
        [
            _StatusResponse(status_code=200, content=b"<html>bad</html>", content_type="text/html"),
            _StatusResponse(status_code=200, content=b"abc", content_type="image/jpeg"),
        ]
    )
    svc = DownloadService()

    async def _fake_get_session():
        return dummy

    async def _run():
        await config.load()
        svc._get_session = _fake_get_session  # type: ignore[method-assign]
        svc._build_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        return await svc.to_base64("/imagine-public/images/demo.jpg", "token")

    result = asyncio.run(_run())

    assert dummy.urls == [
        "https://imagine-public.x.ai/imagine-public/images/demo.jpg",
        "https://grok.com/imagine-public/images/demo.jpg",
    ]
    assert result.startswith("data:image/jpeg;base64,")


def test_ws_download_keeps_absolute_url_host():
    dummy = _DummyDownloadService()
    processor = ImageWSCollectProcessor("grok-imagine-1.0", "token", n=1)
    processor._get_dl = lambda: dummy  # type: ignore[method-assign]

    async def _run():
        await config.load()
        return await processor._download_from_url(
            "https://imagine-public.x.ai/imagine-public/images/demo.jpg", "img-1"
        )

    result = asyncio.run(_run())

    assert result == "AAA"
    assert dummy.calls == [
        "https://imagine-public.x.ai/imagine-public/images/demo.jpg"
    ]
