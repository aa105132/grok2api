"""
Chat 图片生成与编辑服务
"""

import re
import time
import uuid
from typing import Any, Dict, List

import orjson

from app.core.config import get_config
from app.core.exceptions import (
    AppException,
    ErrorType,
    UpstreamException,
    ValidationException,
)
from app.core.logger import logger
from app.services.grok.models.model import ModelService
from app.services.grok.processors import ImageCollectProcessor, ImageStreamProcessor
from app.services.grok.processors.image_ws_processors import (
    ImageWSCollectProcessor,
    ImageWSStreamProcessor,
)
from app.services.grok.services.assets import UploadService
from app.services.grok.services.chat import GrokChatService, MessageExtractor
from app.services.grok.services.image import image_service
from app.services.grok.services.media import VideoService
from app.services.grok.utils.stream import wrap_stream_with_usage
from app.services.token import EffortType, get_token_manager


class ImageChatService:
    """Chat 图片模型服务"""

    @staticmethod
    async def completions(
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = None,
        thinking: str = None,
        n: int = 1,
    ):
        """Chat 图片模型入口（类似 VideoService.completions）"""
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()

        is_stream = stream if stream is not None else get_config("chat.stream")
        _ = {"enabled": True, "disabled": False}.get(thinking)

        tried_tokens = set()
        max_token_retries = int(get_config("retry.max_retry"))
        last_error = None

        async def _pick_token():
            token = None
            for pool_name in ModelService.pool_candidates_for_model(model):
                token = token_mgr.get_token(pool_name, exclude=tried_tokens)
                if token:
                    break

            if not token and not tried_tokens:
                logger.info("No available tokens, attempting to refresh cooling tokens...")
                result = await token_mgr.refresh_cooling_tokens()
                if result.get("recovered", 0) > 0:
                    for pool_name in ModelService.pool_candidates_for_model(model):
                        token = token_mgr.get_token(pool_name)
                        if token:
                            break

            return token

        if is_stream:
            async def stream_with_retry():
                nonlocal last_error
                emitted_any = False

                for attempt in range(max_token_retries):
                    token = await _pick_token()
                    if not token:
                        if last_error:
                            raise last_error
                        raise AppException(
                            message="No available tokens. Please try again later.",
                            error_type=ErrorType.RATE_LIMIT.value,
                            code="rate_limit_exceeded",
                            status_code=429,
                        )

                    tried_tokens.add(token)

                    try:
                        if model == "grok-imagine-1.0":
                            result_stream = await ImageChatService.generate_image(
                                model=model,
                                is_stream=True,
                                token=token,
                                messages=messages,
                                token_mgr=token_mgr,
                                n=n,
                            )
                        elif model == "grok-imagine-1.0-edit":
                            result_stream = await ImageChatService.edit_image(
                                model=model,
                                is_stream=True,
                                token=token,
                                messages=messages,
                                token_mgr=token_mgr,
                            )
                        else:
                            raise ValidationException(f"Unsupported image model: {model}")

                        async for chunk in result_stream:
                            emitted_any = True
                            yield chunk
                        return

                    except UpstreamException as e:
                        status_code = e.details.get("status") if e.details else None
                        last_error = e

                        # 已经向客户端输出数据后，不再切换 token 重试，避免混流
                        if emitted_any:
                            raise

                        if status_code == 429:
                            await token_mgr.mark_rate_limited(token)
                            logger.warning(
                                f"Token {token[:10]}... rate limited (429), "
                                f"trying next token (attempt {attempt + 1}/{max_token_retries})"
                            )
                            continue

                        logger.warning(
                            "Image stream failed before first chunk, "
                            f"trying next token (attempt {attempt + 1}/{max_token_retries}): {e}"
                        )
                        continue

                    except Exception as e:
                        last_error = e

                        if emitted_any:
                            raise

                        logger.warning(
                            "Image stream unexpected error before first chunk, "
                            f"trying next token (attempt {attempt + 1}/{max_token_retries}): {e}"
                        )
                        continue

                if last_error:
                    raise last_error
                raise AppException(
                    message="No available tokens. Please try again later.",
                    error_type=ErrorType.RATE_LIMIT.value,
                    code="rate_limit_exceeded",
                    status_code=429,
                )

            return stream_with_retry()

        for attempt in range(max_token_retries):
            token = await _pick_token()
            if not token:
                if last_error:
                    raise last_error
                raise AppException(
                    message="No available tokens. Please try again later.",
                    error_type=ErrorType.RATE_LIMIT.value,
                    code="rate_limit_exceeded",
                    status_code=429,
                )

            tried_tokens.add(token)

            try:
                if model == "grok-imagine-1.0":
                    return await ImageChatService.generate_image(
                        model=model,
                        is_stream=False,
                        token=token,
                        messages=messages,
                        token_mgr=token_mgr,
                        n=n,
                    )

                if model == "grok-imagine-1.0-edit":
                    return await ImageChatService.edit_image(
                        model=model,
                        is_stream=False,
                        token=token,
                        messages=messages,
                        token_mgr=token_mgr,
                    )

                raise ValidationException(f"Unsupported image model: {model}")

            except UpstreamException as e:
                status_code = e.details.get("status") if e.details else None
                last_error = e

                if status_code == 429:
                    await token_mgr.mark_rate_limited(token)
                    logger.warning(
                        f"Token {token[:10]}... rate limited (429), "
                        f"trying next token (attempt {attempt + 1}/{max_token_retries})"
                    )
                    continue

                raise

        if last_error:
            raise last_error
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    @staticmethod
    def _render_markdown_images(data: Dict[str, Any]) -> str:
        """将图片响应转换为 markdown 图片列表"""
        content = ""
        if b64 := data.get("b64_json"):
            img_id = str(uuid.uuid4())[:8]
            content += f"![{img_id}](data:image/jpeg;base64,{b64})\n"
        if url := data.get("url"):
            img_id = str(uuid.uuid4())[:8]
            content += f"![{img_id}]({url})\n"
        if b64 := data.get("base64"):
            img_id = str(uuid.uuid4())[:8]
            content += f"![{img_id}](data:image/jpeg;base64,{b64})\n"
        if not content and "error" in data:
            content += f"Error: {data['error']}\n"
        return content

    @staticmethod
    async def generate_image(
        model: str,
        is_stream: bool,
        token: str,
        messages: List[Dict[str, Any]],
        token_mgr,
        n: int = 1,
    ):
        """图片生成处理函数"""
        message, _ = MessageExtractor.extract(messages)
        gen_request = image_service.stream(token, message, n=n)

        if is_stream:
            logger.debug(f"Processing image stream response: model={model}")
            image_format = get_config("app.image_format", "b64_json")
            processor = ImageWSStreamProcessor(model, token, n, image_format)

            async def chat_stream_wrapper():
                role_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {orjson.dumps(role_chunk).decode()}\n\n"

                async for sse_msg in processor.process(gen_request):
                    if not sse_msg.strip():
                        continue

                    if sse_msg.startswith("event: image_generation.completed"):
                        try:
                            data_line = ""
                            for line in sse_msg.splitlines():
                                if line.startswith("data: "):
                                    data_line = line[6:].strip()
                                    break

                            if not data_line:
                                continue

                            data = orjson.loads(data_line)
                            content = ImageChatService._render_markdown_images(data)

                            if not content:
                                logger.warning(f"Invalid image response: {data}")
                                continue

                            chunk = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to process image SSE: {e}")
                    elif sse_msg.startswith("event: image_generation.partial_image"):
                        pass
                    elif sse_msg.startswith("event: error"):
                        yield sse_msg

                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {orjson.dumps(final_chunk).decode()}\n\n"
                yield "data: [DONE]\n\n"

            return wrap_stream_with_usage(chat_stream_wrapper(), token_mgr, token, model)

        logger.debug(f"Processing image collect response: model={model}")
        image_format = get_config("app.image_format", "b64_json")
        processor = ImageWSCollectProcessor(model, token, n, image_format)
        image_results = await processor.process(gen_request)

        content = ""
        for img_data in image_results:
            img_id = str(uuid.uuid4())[:8]
            if img_data.startswith("http") or img_data.startswith("/v1/files"):
                content += f"![{img_id}]({img_data})\n"
            else:
                content += f"![{img_id}](data:image/jpeg;base64,{img_data})\n"

        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        try:
            effort = EffortType.HIGH
            await token_mgr.consume(token, effort)
            logger.info(f"Image chat completed: model={model}, effort={effort.value}")
        except Exception as e:
            logger.warning(f"Failed to record usage: {e}")

        return result

    @staticmethod
    async def edit_image(
        model: str,
        is_stream: bool,
        token: str,
        messages: List[Dict[str, Any]],
        token_mgr,
    ):
        """图片编辑处理函数"""
        prompt, attachments = MessageExtractor.extract(messages)

        image_urls: List[str] = []
        upload_service = UploadService()
        try:
            for attach_type, attach_data in attachments:
                if attach_type == "image":
                    _, file_uri = await upload_service.upload(attach_data, token)
                    if file_uri:
                        if file_uri.startswith("http"):
                            image_urls.append(file_uri)
                        else:
                            image_urls.append(
                                f"https://assets.grok.com/{file_uri.lstrip('/')}"
                            )
        finally:
            await upload_service.close()

        if not image_urls:
            raise ValidationException("No image provided for editing")

        parent_post_id = None
        try:
            media_service = VideoService()
            parent_post_id = await media_service.create_image_post(token, image_urls[0])
        except Exception as e:
            logger.warning(f"Create image post failed: {e}")

        if not parent_post_id:
            for url in image_urls:
                match = re.search(r"/generated/([a-f0-9-]+)/", url)
                if match:
                    parent_post_id = match.group(1)
                    break
                match = re.search(r"/users/[^/]+/([a-f0-9-]+)/content", url)
                if match:
                    parent_post_id = match.group(1)
                    break

        model_info = ModelService.get(model)
        model_config_override = {
            "modelMap": {
                "imageEditModel": "imagine",
                "imageEditModelConfig": {
                    "imageReferences": image_urls,
                },
            }
        }
        if parent_post_id:
            model_config_override["modelMap"]["imageEditModelConfig"][
                "parentPostId"
            ] = parent_post_id

        raw_payload = {
            "temporary": bool(get_config("chat.temporary")),
            "modelName": model_info.grok_model,
            "message": prompt,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {"imageGen": True},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "disableTextFollowUps": True,
            "responseMetadata": {"modelConfigOverride": model_config_override},
            "disableMemory": False,
            "forceSideBySide": False,
        }

        service = GrokChatService()
        response = await service.chat(
            token=token,
            message=prompt,
            model=model_info.grok_model,
            mode=None,
            stream=True,
            raw_payload=raw_payload,
        )

        image_format = get_config("app.image_format")
        if is_stream:
            logger.debug(f"Processing image edit stream response: model={model}")
            processor = ImageStreamProcessor(model, token, n=1, response_format=image_format)

            async def chat_stream_wrapper():
                role_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {orjson.dumps(role_chunk).decode()}\n\n"

                async for sse_msg in processor.process(response):
                    if not sse_msg.strip():
                        continue

                    if sse_msg.startswith("event: image_generation.completed"):
                        try:
                            data_line = ""
                            for line in sse_msg.splitlines():
                                if line.startswith("data: "):
                                    data_line = line[6:].strip()
                                    break

                            if not data_line:
                                continue

                            data = orjson.loads(data_line)
                            content = ImageChatService._render_markdown_images(data)

                            if not content:
                                logger.warning(f"Invalid image response: {data}")
                                continue

                            chunk = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to process image edit SSE: {e}")
                    elif sse_msg.startswith("event: error"):
                        yield sse_msg

                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {orjson.dumps(final_chunk).decode()}\n\n"
                yield "data: [DONE]\n\n"

            return wrap_stream_with_usage(chat_stream_wrapper(), token_mgr, token, model)

        logger.debug(f"Processing image edit collect response: model={model}")
        processor = ImageCollectProcessor(model, token, response_format=image_format)
        image_results = await processor.process(response)
        
        content = ""
        for img_data in image_results:
            img_id = str(uuid.uuid4())[:8]
            if img_data.startswith("http") or img_data.startswith("/v1/files"):
                content += f"![{img_id}]({img_data})\n"
            else:
                content += f"![{img_id}](data:image/jpeg;base64,{img_data})\n"
        
        if content:
            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content.strip(),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "图片生成失败，可能是被审查拦截了",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        try:
            effort = EffortType.HIGH
            await token_mgr.consume(token, effort)
            logger.info(f"Image edit chat completed: model={model}, effort={effort.value}")
        except Exception as e:
            logger.warning(f"Failed to record usage: {e}")

        return result


__all__ = ["ImageChatService"]
