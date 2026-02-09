"""
图片生成响应处理器（WebSocket）
"""

import time
from typing import AsyncGenerator, AsyncIterable, List, Dict, Optional

import orjson

from app.core.config import get_config
from app.core.logger import logger
from app.core.exceptions import UpstreamException
from .base import BaseProcessor


class ImageWSBaseProcessor(BaseProcessor):
    """WebSocket 图片处理基类"""

    def __init__(self, model: str, token: str = "", response_format: str = "b64_json"):
        super().__init__(model, token)
        self.response_format = response_format
        if response_format == "url":
            self.response_field = "url"
        elif response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"

    def _strip_base64(self, blob: str) -> str:
        if not blob:
            return ""
        if "," in blob and "base64" in blob.split(",", 1)[0]:
            return blob.split(",", 1)[1]
        return blob

    def _pick_best(self, existing: Optional[Dict], incoming: Dict) -> Dict:
        if not existing:
            if "blob" in incoming and "blob_size" not in incoming:
                incoming["blob_size"] = len(incoming["blob"])
            return incoming
        
        # 获取 blob 大小（如果不存在则计算）
        def get_size(d):
            if "blob_size" in d:
                return d["blob_size"]
            size = len(d.get("blob", ""))
            d["blob_size"] = size
            return size

        def merge_url(target: Dict, source: Dict) -> Dict:
            """确保结果中保留最佳 URL 信息（用于 fallback 下载高质量图片）
            
            优先使用 final 阶段的 URL（指向高质量图片），
            其次使用任何可用的 URL。
            """
            source_url = source.get("url", "")
            target_url = target.get("url", "")
            
            # 如果 source 有 URL 且 source 是 final，优先使用 source 的 URL
            if source_url and source.get("is_final"):
                target["url"] = source_url
            # 如果 target 没有 URL，使用 source 的
            elif not target_url and source_url:
                target["url"] = source_url
            return target

        # 如果 incoming 是 final 但没有 blob，尝试保留 existing 的 blob 并标记为 final
        if incoming.get("is_final") and not incoming.get("blob") and existing.get("blob"):
            res = existing.copy()
            res["is_final"] = True
            return merge_url(res, incoming)

        if incoming.get("is_final") and not existing.get("is_final"):
            # 只有当 incoming 有内容时才替换，否则只更新标记
            if incoming.get("blob"):
                if "blob_size" not in incoming:
                    incoming["blob_size"] = len(incoming["blob"])
                return merge_url(incoming, existing)
            res = existing.copy()
            res["is_final"] = True
            return merge_url(res, incoming)

        if existing.get("is_final") and not incoming.get("is_final"):
            # 保留 existing，但如果 incoming 有更新的 url，也合并进来
            return merge_url(existing, incoming)
        
        if get_size(incoming) > get_size(existing):
            return merge_url(incoming, existing)
        return merge_url(existing, incoming)

    async def _to_output(self, image_id: str, item: Dict) -> str:
        try:
            blob = item.get("blob", "")
            url = item.get("url", "")
            is_final = item.get("is_final", False)

            # 判断 blob 是否为 final 质量
            # 如果 blob 存在但不满足 final 大小阈值，说明是 medium/preview 质量
            blob_size = len(blob) if blob else 0
            final_min_bytes = get_config("image.image_ws_final_min_bytes")
            blob_is_final_quality = blob_size >= final_min_bytes

            # 对于 final 图片，如果 blob 不是 final 质量，优先通过 URL 下载高质量版本
            if is_final and url and not blob_is_final_quality:
                logger.info(
                    f"Final image {image_id} has low-quality blob ({blob_size} bytes), "
                    f"downloading HD from URL: {url[:80]}"
                )
                hd_b64 = await self._download_from_url(url, image_id)
                if hd_b64:
                    if self.response_format == "url":
                        # 直接返回 Grok 资源 URL，不保存本地
                        return url if url.startswith("http") else f"https://assets.grok.com{url}"
                    return hd_b64

            if self.response_format == "url":
                # URL 模式：直接返回 Grok 资源 URL，不保存本地
                if url:
                    return url if url.startswith("http") else f"https://assets.grok.com{url}"
                # 如果没有 URL 但有 blob，无法提供 URL，返回空字符串
                logger.warning(f"No URL available for image {image_id} in url mode")
                return ""

            # b64_json / base64 模式：返回 base64 数据
            b64 = self._strip_base64(blob)
            if b64:
                return b64
            # blob 为空，尝试通过 URL 下载并转 base64
            if url:
                logger.info(f"Blob empty for {image_id}, downloading from URL: {url[:80]}")
                return await self._download_from_url(url, image_id)
            return ""
        except Exception as e:
            logger.warning(f"Image output failed: {e}")
            return ""

    async def _download_from_url(self, url: str, image_id: str) -> str:
        """异步下载 URL 并转 base64（用于 blob 为空时 fallback）"""
        try:
            dl_service = self._get_dl()
            # 构建路径
            if url.startswith("http"):
                from urllib.parse import urlparse
                path = urlparse(url).path
            else:
                path = url

            base64_data = await dl_service.to_base64(path, self.token, "image")
            if base64_data:
                if "," in base64_data:
                    return base64_data.split(",", 1)[1]
                return base64_data
            return ""
        except Exception as e:
            logger.warning(f"Download fallback failed for {image_id}: {e}")
            return ""


class ImageWSStreamProcessor(ImageWSBaseProcessor):
    """WebSocket 图片流式响应处理器"""

    def __init__(
        self,
        model: str,
        token: str = "",
        n: int = 1,
        response_format: str = "b64_json",
        size: str = "1024x1024",
    ):
        super().__init__(model, token, response_format)
        self.n = n
        self.size = size
        self._target_id: Optional[str] = None
        self._index_map: Dict[str, int] = {}
        self._partial_map: Dict[str, int] = {}

    def _assign_index(self, image_id: str) -> Optional[int]:
        if image_id in self._index_map:
            return self._index_map[image_id]
        
        # 即使超过 n，也记录 image_id，但返回 None 表示不处理流式部分
        idx = len(self._index_map)
        self._index_map[image_id] = idx
        
        if idx >= self.n:
            return None
        return idx

    def _sse(self, event: str, data: dict) -> str:
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"

    async def process(self, response: AsyncIterable[dict]) -> AsyncGenerator[str, None]:
        images: Dict[str, Dict] = {}

        async for item in response:
            if item.get("type") == "error":
                message = item.get("error") or "Upstream error"
                code = item.get("error_code") or "upstream_error"
                yield self._sse(
                    "error",
                    {
                        "error": {
                            "message": message,
                            "type": "server_error",
                            "code": code,
                        }
                    },
                )
                return
            if item.get("type") != "image":
                continue

            image_id = item.get("image_id")
            if not image_id:
                continue

            # 统一 is_final 标记
            if item.get("stage") == "final":
                item["is_final"] = True

            if self.n == 1:
                if self._target_id is None:
                    self._target_id = image_id
                index = 0 if image_id == self._target_id else None
            else:
                index = self._assign_index(image_id)

            images[image_id] = self._pick_best(images.get(image_id), item)

            if index is None:
                continue

            if item.get("stage") != "final":
                partial_b64 = self._strip_base64(item.get("blob", ""))
                if not partial_b64:
                    continue
                partial_index = self._partial_map.get(image_id, 0)
                if item.get("stage") == "medium":
                    partial_index = max(partial_index, 1)
                self._partial_map[image_id] = partial_index
                yield self._sse(
                    "image_generation.partial_image",
                    {
                        "type": "image_generation.partial_image",
                        "b64_json": partial_b64,
                        "created_at": int(time.time()),
                        "size": self.size,
                        "index": index,
                        "partial_image_index": partial_index,
                    },
                )

        # 结束流式传输，发送最终结果
        # 从所有收到的图片中选出最好的 n 张
        all_images = sorted(
            images.values(),
            key=lambda x: (x.get("is_final", False), x.get("blob_size", 0)),
            reverse=True,
        )
        selected_items = all_images[: self.n]

        for i, item in enumerate(selected_items):
            image_id = item.get("image_id", "")
            output = await self._to_output(image_id, item)
            if not output:
                continue

            res_data = {
                "type": "image_generation.completed",
                "created_at": int(time.time()),
                "size": self.size,
                "index": i,
                "usage": {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
                },
            }
            res_data[self.response_field] = output
            
            yield self._sse("image_generation.completed", res_data)


class ImageWSCollectProcessor(ImageWSBaseProcessor):
    """WebSocket 图片非流式响应处理器"""

    def __init__(
        self, model: str, token: str = "", n: int = 1, response_format: str = "b64_json"
    ):
        super().__init__(model, token, response_format)
        self.n = n
    
    async def process(self, response: AsyncIterable[dict]) -> List[str]:
        images: Dict[str, Dict] = {}

        async for item in response:
            if item.get("type") == "error":
                message = item.get("error") or "Upstream error"
                raise UpstreamException(message, details=item)
            if item.get("type") != "image":
                continue
            image_id = item.get("image_id")
            if not image_id:
                continue
            
            # 统一 is_final 标记
            if item.get("stage") == "final":
                item["is_final"] = True
                
            images[image_id] = self._pick_best(images.get(image_id), item)

        selected = sorted(
            images.values(),
            key=lambda x: (x.get("is_final", False), x.get("blob_size", 0)),
            reverse=True,
        )
        if self.n:
            selected = selected[: self.n]

        results: List[str] = []
        for item in selected:
            output = await self._to_output(item.get("image_id", ""), item)
            if output:
                results.append(output)

        return results


__all__ = ["ImageWSStreamProcessor", "ImageWSCollectProcessor"]
