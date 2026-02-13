from app.services.grok.services.chat import _build_chat_image_markdown


def test_build_chat_image_markdown_from_b64_json():
    content = _build_chat_image_markdown({"b64_json": "AAA"})
    assert "(data:image/jpeg;base64,AAA)" in content


def test_build_chat_image_markdown_from_base64_alias():
    content = _build_chat_image_markdown({"base64": "BBB"})
    assert "(data:image/jpeg;base64,BBB)" in content


def test_build_chat_image_markdown_base64_field_with_url_fallback():
    content = _build_chat_image_markdown(
        {"base64": "https://assets.grok.com/users/u1/generated/a1/image.jpg"}
    )
    assert "(https://assets.grok.com/users/u1/generated/a1/image.jpg)" in content


def test_build_chat_image_markdown_from_url():
    content = _build_chat_image_markdown({"url": "/v1/files/image/demo.jpg"})
    assert "(/v1/files/image/demo.jpg)" in content
