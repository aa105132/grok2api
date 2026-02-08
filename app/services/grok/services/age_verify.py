"""
年龄验证服务

在首次使用时自动完成成人年龄认证，供图像生成等功能使用。
"""

from dataclasses import dataclass
from typing import Optional
import datetime
import random

from curl_cffi.requests import AsyncSession

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.utils.headers import build_sso_cookie

BIRTH_DATE_API = "https://grok.com/rest/auth/set-birth-date"


@dataclass
class AgeVerifyResult:
    """年龄验证结果"""

    success: bool
    http_status: int
    error: Optional[str] = None


class AgeVerifyService:
    """年龄验证服务
    
    首次使用时自动设置出生日期完成年龄认证。
    """

    def __init__(self, proxy: str = None):
        self.proxy = proxy or get_config("network.base_proxy_url")
        self.timeout = float(get_config("network.timeout"))

    def _build_proxies(self) -> Optional[dict]:
        """构建代理配置"""
        return {"http": self.proxy, "https": self.proxy} if self.proxy else None

    @staticmethod
    def _random_birth_date() -> str:
        """生成随机出生日期（20-40岁）"""
        today = datetime.date.today()
        birth_year = today.year - random.randint(20, 40)
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        microsecond = random.randint(0, 999)
        return f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{microsecond:03d}Z"

    def _build_headers(self, token: str) -> dict:
        """构造请求头"""
        cookie = build_sso_cookie(token, include_rw=True)
        user_agent = get_config("security.user_agent")
        cf_clearance = get_config("security.cf_clearance")
        
        # 如果有 cf_clearance，添加到 cookie
        if cf_clearance:
            cookie = f"{cookie}; cf_clearance={cf_clearance}"
        
        return {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "referer": "https://grok.com/?_s=account",
            "user-agent": user_agent,
            "cookie": cookie,
        }

    async def verify(self, token: str) -> AgeVerifyResult:
        """为单个 token 完成年龄验证"""
        headers = self._build_headers(token)
        payload = {"birthDate": self._random_birth_date()}
        
        logger.info(f"Age verification: starting for token {token[:10]}...")

        try:
            browser = get_config("security.browser")
            async with AsyncSession(impersonate=browser) as session:
                response = await session.post(
                    BIRTH_DATE_API,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=self._build_proxies(),
                )

                if response.status_code in (200, 204):
                    logger.info(f"Age verification: success for token {token[:10]}...")
                    return AgeVerifyResult(
                        success=True,
                        http_status=response.status_code,
                    )
                
                error_msg = f"HTTP {response.status_code}"
                logger.warning(f"Age verification: failed for token {token[:10]}... - {error_msg}")
                return AgeVerifyResult(
                    success=False,
                    http_status=response.status_code,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"Age verification: exception for token {token[:10]}... - {e}")
            return AgeVerifyResult(
                success=False,
                http_status=0,
                error=str(e)[:100],
            )


# 全局实例
age_verify_service = AgeVerifyService()


async def ensure_age_verified(token: str, token_mgr) -> bool:
    """
    确保 token 已完成年龄验证
    
    Args:
        token: SSO token
        token_mgr: TokenManager 实例
    
    Returns:
        是否已验证或验证成功
    """
    # 检查当前验证状态
    age_status = await token_mgr.get_age_verified(token)
    
    if age_status == 1:
        # 已验证
        logger.debug(f"Token {token[:10]}... already age verified")
        return True
    
    if age_status == 2:
        # 之前验证失败，不再重试
        logger.debug(f"Token {token[:10]}... previously failed age verification")
        return False
    
    # 未验证，执行验证
    logger.info(f"Token {token[:10]}... needs age verification")
    result = await age_verify_service.verify(token)
    
    if result.success:
        await token_mgr.set_age_verified(token, 1)
        return True
    else:
        await token_mgr.set_age_verified(token, 2)
        return False


__all__ = ["AgeVerifyService", "AgeVerifyResult", "age_verify_service", "ensure_age_verified"]