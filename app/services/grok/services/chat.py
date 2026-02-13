"""
Grok Chat 服务
"""

import re
import time
import uuid
import orjson
from typing import Dict, List, Any
from dataclasses import dataclass

from curl_cffi.requests import AsyncSession

from app.core.logger import logger
from app.core.config import get_config
from app.core.exceptions import (
    AppException,
    UpstreamException,
    ValidationException,
    ErrorType,
)
from app.services.grok.models.model import ModelService
from app.services.grok.services.assets import UploadService
from app.services.grok.processors import StreamProcessor, CollectProcessor
from app.services.grok.utils.retry import retry_on_status
from app.services.grok.utils.headers import apply_statsig, build_sso_cookie
from app.services.grok.utils.stream import wrap_stream_with_usage
from app.services.token import get_token_manager, EffortType, TokenManager

from app.services.grok.services.image import image_service
from app.services.grok.processors.image_ws_processors import ImageWSStreamProcessor, ImageWSCollectProcessor

from app.services.grok.services.media import VideoService
from app.services.grok.processors import ImageStreamProcessor, ImageCollectProcessor

CHAT_API = "https://grok.com/rest/app-chat/conversations/new"


def _extract_sse_data_payload(sse_msg: str) -> Dict[str, Any]:
    """从 SSE 文本中提取首个 data JSON。"""
    data_line = ""
    for line in sse_msg.splitlines():
        if line.startswith("data: "):
            data_line = line[6:].strip()
            break

    if not data_line:
        return {}

    try:
        payload = orjson.loads(data_line)
    except orjson.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _build_chat_image_markdown(payload: Dict[str, Any]) -> str:
    """将图片 payload 统一转换为 chat markdown。"""
    raw = payload.get("b64_json")
    if raw is None:
        raw = payload.get("base64")

    if raw:
        value = str(raw).strip()
        if not value:
            return ""
        img_id = str(uuid.uuid4())[:8]
        # base64 字段偶尔会回退成 URL，这里统一兜底处理。
        if value.startswith("http") or value.startswith("/") or value.startswith("data:"):
            return f"![{img_id}]({value})\n"
        return f"![{img_id}](data:image/jpeg;base64,{value})\n"

    url = payload.get("url")
    if url:
        url_value = str(url).strip()
        if url_value:
            img_id = str(uuid.uuid4())[:8]
            return f"![{img_id}]({url_value})\n"

    return ""


@dataclass
class ChatRequest:
    """聊天请求数据"""

    model: str
    messages: List[Dict[str, Any]]
    stream: bool = None
    think: bool = None


class MessageExtractor:
    """消息内容提取器"""

    @staticmethod
    def extract(
        messages: List[Dict[str, Any]], is_video: bool = False
    ) -> tuple[str, List[tuple[str, str]]]:
        """从 OpenAI 消息格式提取内容，返回 (text, attachments)"""
        texts = []
        attachments = []
        extracted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts = []

            if isinstance(content, str):
                if content.strip():
                    parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type", "")

                    if item_type == "text":
                        if text := item.get("text", "").strip():
                            parts.append(text)

                    elif item_type == "image_url":
                        image_data = item.get("image_url", {})
                        url = (
                            image_data.get("url", "")
                            if isinstance(image_data, dict)
                            else str(image_data)
                        )
                        if url:
                            attachments.append(("image", url))

                    elif item_type == "input_audio":
                        if is_video:
                            raise ValueError("视频模型不支持 input_audio 类型")
                        audio_data = item.get("input_audio", {})
                        data = (
                            audio_data.get("data", "")
                            if isinstance(audio_data, dict)
                            else str(audio_data)
                        )
                        if data:
                            attachments.append(("audio", data))

                    elif item_type == "file":
                        if is_video:
                            raise ValueError("视频模型不支持 file 类型")
                        file_data = item.get("file", {})
                        url = file_data.get("url", "") or file_data.get("data", "")
                        if isinstance(file_data, str):
                            url = file_data
                        if url:
                            attachments.append(("file", url))

            if parts:
                extracted.append({"role": role, "text": "\n".join(parts)})

        # 找到最后一条 user 消息
        last_user_index = next(
            (
                i
                for i in range(len(extracted) - 1, -1, -1)
                if extracted[i]["role"] == "user"
            ),
            None,
        )

        for i, item in enumerate(extracted):
            role = item["role"] or "user"
            text = item["text"]
            texts.append(text if i == last_user_index else f"{role}: {text}")

        return "\n\n".join(texts), attachments


class ChatRequestBuilder:
    """请求构造器"""

    @staticmethod
    def build_headers(token: str) -> Dict[str, str]:
        """构造请求头"""
        user_agent = get_config("security.user_agent")
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Baggage": "sentry-environment=production,sentry-release=d6add6fb0460641fd482d767a335ef72b9b6abb8,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Origin": "https://grok.com",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Referer": "https://grok.com/",
            "Sec-Ch-Ua": '"Google Chrome";v="136", "Chromium";v="136", "Not(A:Brand";v="24"',
            "Sec-Ch-Ua-Arch": "arm",
            "Sec-Ch-Ua-Bitness": "64",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Model": "",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": user_agent,
        }

        apply_statsig(headers)
        headers["Cookie"] = build_sso_cookie(token)

        return headers

    @staticmethod
    def build_payload(
        message: str,
        model: str,
        mode: str = None,
        file_attachments: List[str] = None,
        image_attachments: List[str] = None,
    ) -> Dict[str, Any]:
        """构造请求体"""
        merged_attachments = []
        if file_attachments:
            merged_attachments.extend(file_attachments)
        if image_attachments:
            merged_attachments.extend(image_attachments)

        payload = {
            "temporary": get_config("chat.temporary"),
            "modelName": model,
            "message": message,
            "fileAttachments": merged_attachments,
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "responseMetadata": {
                "modelConfigOverride": {"modelMap": {}},
                "requestModelDetails": {"modelId": model},
            },
            "disableMemory": get_config("chat.disable_memory"),
            "deviceEnvInfo": {
                "darkModeEnabled": False,
                "devicePixelRatio": 2,
                "screenWidth": 2056,
                "screenHeight": 1329,
                "viewportWidth": 2056,
                "viewportHeight": 1083,
            },
        }

        if mode:
            payload["modelMode"] = mode

        return payload


class GrokChatService:
    """Grok API 调用服务"""

    def __init__(self, proxy: str = None):
        self.proxy = proxy or get_config("network.base_proxy_url")

    async def chat(
        self,
        token: str,
        message: str,
        model: str = "grok-3",
        mode: str = None,
        stream: bool = None,
        file_attachments: List[str] = None,
        image_attachments: List[str] = None,
        raw_payload: Dict[str, Any] = None,
    ):
        """发送聊天请求"""
        if stream is None:
            stream = get_config("chat.stream")

        headers = ChatRequestBuilder.build_headers(token)
        payload = (
            raw_payload
            if raw_payload is not None
            else ChatRequestBuilder.build_payload(
                message, model, mode, file_attachments, image_attachments
            )
        )
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        timeout = get_config("network.timeout")

        # Log the actual payload being sent
        if raw_payload is not None:
            logger.info(f"Chat raw_payload: {orjson.dumps(payload).decode()[:500]}")

        logger.debug(
            f"Chat request: model={model}, mode={mode}, stream={stream}, attachments={len(file_attachments or [])}"
        )

        # 建立连接
        async def establish_connection():
            browser = get_config("security.browser")
            session = AsyncSession(impersonate=browser)
            try:
                response = await session.post(
                    CHAT_API,
                    headers=headers,
                    data=orjson.dumps(payload),
                    timeout=timeout,
                    stream=True,
                    proxies=proxies,
                )

                if response.status_code != 200:
                    content = ""
                    try:
                        content = await response.text()
                    except Exception:
                        pass

                    logger.error(
                        f"Chat failed: status={response.status_code}, token={token[:10]}..."
                    )

                    await session.close()
                    raise UpstreamException(
                        message=f"Grok API request failed: {response.status_code}",
                        details={"status": response.status_code, "body": content},
                    )

                logger.info(f"Chat connected: model={model}, stream={stream}")
                return session, response

            except UpstreamException:
                raise
            except Exception as e:
                logger.error(f"Chat request error: {e}")
                await session.close()
                raise UpstreamException(
                    message=f"Chat connection failed: {str(e)}",
                    details={"error": str(e)},
                )

        # 重试机制
        def extract_status(e: Exception) -> int | None:
            if isinstance(e, UpstreamException) and e.details:
                status = e.details.get("status")
                # 429 不在内层重试，由外层跨 token 重试处理
                if status == 429:
                    return None
                return status
            return None

        session = None
        response = None
        try:
            session, response = await retry_on_status(
                establish_connection, extract_status=extract_status
            )
        except Exception as e:
            status_code = extract_status(e)
            if status_code:
                token_mgr = await get_token_manager()
                reason = str(e)
                if isinstance(e, UpstreamException) and e.details:
                    body = e.details.get("body")
                    if body:
                        reason = f"{reason} | body: {body}"
                await token_mgr.record_fail(token, status_code, reason)
            raise

        # 流式传输
        async def stream_response():
            try:
                async for line in response.aiter_lines():
                    yield line
            finally:
                if session:
                    await session.close()

        return stream_response()

    async def chat_openai(self, token: str, request: ChatRequest):
        """OpenAI 兼容接口"""
        model_info = ModelService.get(request.model)
        if not model_info:
            raise ValidationException(f"Unknown model: {request.model}")

        grok_model = model_info.grok_model
        mode = model_info.model_mode
        is_video = model_info.is_video

        # 提取消息和附件
        try:
            message, attachments = MessageExtractor.extract(
                request.messages, is_video=is_video
            )
            logger.debug(
                f"Extracted message length={len(message)}, attachments={len(attachments)}"
            )
        except ValueError as e:
            raise ValidationException(str(e))

        # 上传附件
        file_ids = []
        if attachments:
            upload_service = UploadService()
            try:
                for attach_type, attach_data in attachments:
                    file_id, _ = await upload_service.upload(attach_data, token)
                    file_ids.append(file_id)
                    logger.debug(
                        f"Attachment uploaded: type={attach_type}, file_id={file_id}"
                    )
            finally:
                await upload_service.close()

        stream = (
            request.stream if request.stream is not None else get_config("chat.stream")
        )

        response = await self.chat(
            token,
            message,
            grok_model,
            mode,
            stream,
            file_attachments=file_ids,
            image_attachments=[],
        )

        return response, stream, request.model


class ChatService:
    """Chat 业务服务"""

    @staticmethod
    async def completions(
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = None,
        thinking: str = None,
        n: int = 1,
    ):
        """Chat Completions 入口"""
        # 获取 token
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()

        token = None
        for pool_name in ModelService.pool_candidates_for_model(model):
            token = token_mgr.get_token(pool_name)
            if token:
                break

        if not token:
            raise AppException(
                message="No available tokens. Please try again later.",
                error_type=ErrorType.RATE_LIMIT.value,
                code="rate_limit_exceeded",
                status_code=429,
            )

        # 自动开启 NSFW 模式（首次使用时触发，后台执行不阻塞）
        if get_config("chat.auto_nsfw"):
            try:
                from app.services.grok.services.nsfw import ensure_nsfw_enabled
                await ensure_nsfw_enabled(token, token_mgr)
            except Exception as e:
                logger.warning(f"Auto NSFW enable failed: {e}")

        # 解析参数
        think = {"enabled": True, "disabled": False}.get(thinking)
        is_stream = stream if stream is not None else get_config("chat.stream")

        # 构造请求（只需构造一次）
        chat_request = ChatRequest(
            model=model, messages=messages, stream=is_stream, think=think
        )

        # 请求 Grok
        service = GrokChatService()
        
        # 特殊处理 grok-imagine-1.0
        model_info = ModelService.get(model)
        if model_info and model_info.is_image:
            if model == "grok-imagine-1.0":
                return await ChatService.generate_image(model, is_stream, token, messages, token_mgr, n)
            elif model == "grok-imagine-1.0-edit":
                return await ChatService.edit_image(model, is_stream, token, messages, token_mgr)

        response, _, model_name = await service.chat_openai(token, chat_request)

        for attempt in range(max_token_retries):
            # 选择 token（排除已失败的）
            token = None
            for pool_name in ModelService.pool_candidates_for_model(model):
                token = token_mgr.get_token(pool_name, exclude=tried_tokens)
                if token:
                    break

        # 非流式
        logger.debug(f"Processing non-stream response: model={model}")
        result = await CollectProcessor(model_name, token).process(response)
        try:
            model_info = ModelService.get(model)
            effort = (
                EffortType.HIGH
                if (model_info and model_info.cost.value == "high")
                else EffortType.LOW
            )
            await token_mgr.consume(token, effort)
            logger.info(f"Chat completed: model={model}, effort={effort.value}")
        except Exception as e:
            logger.warning(f"Failed to record usage: {e}")
        return result
    
    @staticmethod
    async def generate_image(model: str, is_stream: bool, token: str, messages: List[str], token_mgr: TokenManager, n: int = 1):
        # 提取提示词
            message, _ = MessageExtractor.extract(messages)
            
            # 调用图片服务
            gen_request = image_service.stream(token, message, n=n)
            
            if is_stream:
                logger.debug(f"Processing image stream response: model={model}")
                # chat completions 中图片以 markdown 形式内嵌，优先使用 b64_json 确保数据完整
                image_format = get_config("app.image_format") or "b64_json"
                processor = ImageWSStreamProcessor(model, token, n, image_format)
                
                async def chat_stream_wrapper():
                    # 先发送 role
                    role_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {orjson.dumps(role_chunk).decode()}\n\n"

                    async for sse_msg in processor.process(gen_request):
                        if not sse_msg.strip():
                            continue
                        
                        if sse_msg.startswith("event: image_generation.completed"):
                            # 提取图片 payload 并包装成 chat.completion.chunk
                            try:
                                data = _extract_sse_data_payload(sse_msg)
                                if not data:
                                    continue

                                content = _build_chat_image_markdown(data)
                                if content:
                                    chunk = {
                                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": content},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                            except Exception as e:
                                logger.warning(f"Failed to process image SSE: {e}")
                        elif sse_msg.startswith("event: image_generation.partial_image"):
                            # 可选：处理进度显示
                            pass
                        elif sse_msg.startswith("event: error"):
                            yield sse_msg
                    
                    # 发送结束标记
                    final_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {orjson.dumps(final_chunk).decode()}\n\n"
                    yield "data: [DONE]\n\n"

                return wrap_stream_with_usage(
                    chat_stream_wrapper(), token_mgr, token, model
                )
            else:
                logger.debug(f"Processing image collect response: model={model}")
                image_format = get_config("app.image_format") or "b64_json"
                processor = ImageWSCollectProcessor(model, token, n, image_format)
                image_results = await processor.process(gen_request)
                
                # 将图片结果转回 chat 标准输出格式
                content = ""
                for img_data in image_results:
                    img_id = str(uuid.uuid4())[:8]
                    if img_data.startswith("http") or img_data.startswith("/v1/files"):
                        content += f"![{img_id}]({img_data})\n"
                    else:
                        # 假设是 base64
                        content += f"![{img_id}](data:image/jpeg;base64,{img_data})\n"
                
                # 构造类似 CollectProcessor 的返回结构
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
                    }
                }
                
                # 记录消耗
                try:
                    effort = EffortType.HIGH
                    await token_mgr.consume(token, effort)
                    logger.info(f"Image chat completed: model={model}, effort={effort.value}")
                except Exception as e:
                    logger.warning(f"Failed to record usage: {e}")
                
                return result

    @staticmethod
    async def edit_image(model: str, is_stream: bool, token: str, messages: List[Dict[str, Any]], token_mgr: TokenManager):
        """图片编辑处理函数"""

        # 1. 提取提示词和附件
        prompt, attachments = MessageExtractor.extract(messages)
        
        # 2. 上传图片并获取 URL
        image_urls: List[str] = []
        upload_service = UploadService()
        try:
            for attach_type, attach_data in attachments:
                if attach_type == "image":
                    file_id, file_uri = await upload_service.upload(attach_data, token)
                    if file_uri:
                        if file_uri.startswith("http"):
                            image_urls.append(file_uri)
                        else:
                            image_urls.append(f"https://assets.grok.com/{file_uri.lstrip('/')}")
        finally:
            await upload_service.close()

        if not image_urls:
            raise ValidationException("No image provided for editing")

        # 3. 创建图片帖子以获取 parentPostId
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

        # 4. 构造请求载荷
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
            model_config_override["modelMap"]["imageEditModelConfig"]["parentPostId"] = parent_post_id

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

        # 5. 调用 Grok
        service = GrokChatService()
        response = await service.chat(
            token=token,
            message=prompt,
            model=model_info.grok_model,
            mode=None,
            stream=True,
            raw_payload=raw_payload,
        )

        # 6. 处理响应
        image_format = get_config("app.image_format")
        if is_stream:
            logger.debug(f"Processing image edit stream response: model={model}")
            processor = ImageStreamProcessor(model, token, n=1, response_format=image_format)
            
            async def chat_stream_wrapper():
                # 先发送 role
                role_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {orjson.dumps(role_chunk).decode()}\n\n"

                async for sse_msg in processor.process(response):
                    if not sse_msg.strip():
                        continue
                    
                    if sse_msg.startswith("event: image_generation.completed"):
                        try:
                            data = _extract_sse_data_payload(sse_msg)
                            if not data:
                                continue

                            content = _build_chat_image_markdown(data)
                            if content:
                                chunk = {
                                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": content},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to process image edit SSE: {e}")
                    elif sse_msg.startswith("event: error"):
                        yield sse_msg
                
                # 发送结束标记
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {orjson.dumps(final_chunk).decode()}\n\n"
                yield "data: [DONE]\n\n"

            return wrap_stream_with_usage(
                chat_stream_wrapper(), token_mgr, token, model
            )
        else:
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
                }
            }
            
            try:
                effort = EffortType.HIGH
                await token_mgr.consume(token, effort)
                logger.info(f"Image edit chat completed: model={model}, effort={effort.value}")
            except Exception as e:
                logger.warning(f"Failed to record usage: {e}")
            
            return result


__all__ = [
    "GrokChatService",
    "ChatRequest",
    "ChatRequestBuilder",
    "MessageExtractor",
    "ChatService",
]
