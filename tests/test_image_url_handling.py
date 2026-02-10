import asyncio

from app.core.config import config
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
