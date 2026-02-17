"""
Grok 模型管理服务

支持内置模型 + 配置文件自定义模型，配置中的模型可覆盖同 ID 的内置模型。
通过 config.toml 的 [[models]] 段落添加新模型，无需修改代码。
"""

from enum import Enum
from typing import Optional, Tuple, List, Dict
from pydantic import BaseModel, Field

from app.core.exceptions import ValidationException
from app.core.logger import logger


class Tier(str, Enum):
    """模型档位"""

    BASIC = "basic"
    SUPER = "super"


class Cost(str, Enum):
    """计费类型"""

    LOW = "low"
    HIGH = "high"


class ModelInfo(BaseModel):
    """模型信息"""

    model_id: str
    grok_model: str
    model_mode: str
    tier: Tier = Field(default=Tier.BASIC)
    cost: Cost = Field(default=Cost.LOW)
    display_name: str
    description: str = ""
    is_video: bool = False
    is_image: bool = False


class ModelService:
    """
    模型管理服务

    内置模型列表 (BUILTIN_MODELS) 作为基线，配置文件中的自定义模型
    会在启动时合并进来。相同 model_id 时配置文件优先（覆盖内置）。

    使用方法：
    1. 在 config.toml 中添加 [[models]] 段落定义新模型
    2. 应用启动时自动调用 load_from_config() 合并
    3. 运行时可通过 reload() 重新加载
    """

    # 内置模型（基线，始终可用）
    BUILTIN_MODELS = [
        ModelInfo(
            model_id="grok-3",
            grok_model="grok-3",
            model_mode="MODEL_MODE_GROK_3",
            cost=Cost.LOW,
            display_name="GROK-3",
        ),
        ModelInfo(
            model_id="grok-3-mini",
            grok_model="grok-3",
            model_mode="MODEL_MODE_GROK_3_MINI_THINKING",
            cost=Cost.LOW,
            display_name="GROK-3-MINI",
        ),
        ModelInfo(
            model_id="grok-3-thinking",
            grok_model="grok-3",
            model_mode="MODEL_MODE_GROK_3_THINKING",
            cost=Cost.LOW,
            display_name="GROK-3-THINKING",
        ),
        ModelInfo(
            model_id="grok-4",
            grok_model="grok-4",
            model_mode="MODEL_MODE_GROK_4",
            cost=Cost.LOW,
            display_name="GROK-4",
        ),
        ModelInfo(
            model_id="grok-4-mini",
            grok_model="grok-4-mini",
            model_mode="MODEL_MODE_GROK_4_MINI_THINKING",
            cost=Cost.LOW,
            display_name="GROK-4-MINI",
        ),
        ModelInfo(
            model_id="grok-4-thinking",
            grok_model="grok-4",
            model_mode="MODEL_MODE_GROK_4_THINKING",
            cost=Cost.LOW,
            display_name="GROK-4-THINKING",
        ),
        ModelInfo(
            model_id="grok-4-heavy",
            grok_model="grok-4",
            model_mode="MODEL_MODE_HEAVY",
            cost=Cost.HIGH,
            tier=Tier.SUPER,
            display_name="GROK-4-HEAVY",
        ),
        ModelInfo(
            model_id="grok-4.1-mini",
            grok_model="grok-4-1-thinking-1129",
            model_mode="MODEL_MODE_GROK_4_1_MINI_THINKING",
            cost=Cost.LOW,
            display_name="GROK-4.1-MINI",
        ),
        ModelInfo(
            model_id="grok-4.20-fast",
            grok_model="grok-420",
            model_mode="MODEL_MODE_FAST",
            cost=Cost.LOW,
            display_name="GROK-4.20-FAST",
        ),
        ModelInfo(
            model_id="grok-4.20-expert",
            grok_model="grok-420",
            model_mode="MODEL_MODE_EXPERT",
            cost=Cost.HIGH,
            display_name="GROK-4.20-EXPERT",
        ),
        ModelInfo(
            model_id="grok-4.20",
            grok_model="grok-420",
            model_mode="MODEL_MODE_GROK_420",
            cost=Cost.HIGH,
            display_name="GROK-4.20",
        ),
        ModelInfo(
            model_id="grok-4.20-beta",
            grok_model="grok-420",
            model_mode="MODEL_MODE_GROK_420",
            cost=Cost.LOW,
            display_name="GROK-4.20-BETA",
        ),
        ModelInfo(
            model_id="grok-imagine-1.0",
            grok_model="grok-3",
            model_mode="MODEL_MODE_FAST",
            cost=Cost.HIGH,
            display_name="Grok Image",
            description="Image generation model",
            is_image=True,
        ),
        ModelInfo(
            model_id="grok-imagine-1.0-edit",
            grok_model="imagine-image-edit",
            model_mode="MODEL_MODE_FAST",
            cost=Cost.HIGH,
            display_name="Grok Image Edit",
            description="Image edit model",
            is_image=True,
        ),
        ModelInfo(
            model_id="grok-imagine-1.0-video",
            grok_model="grok-3",
            model_mode="MODEL_MODE_FAST",
            cost=Cost.HIGH,
            display_name="Grok Video",
            description="Video generation model",
            is_video=True,
        ),
    ]

    # 兼容旧引用
    MODELS = BUILTIN_MODELS

    _map: Dict[str, ModelInfo] = {m.model_id: m for m in BUILTIN_MODELS}

    @classmethod
    def load_from_config(cls) -> None:
        """
        从配置中加载自定义模型并合并到模型注册表

        配置格式 (config.toml):
            [[models]]
            model_id = "grok-5"
            grok_model = "grok-5"
            model_mode = "MODEL_MODE_GROK_5"
            cost = "low"
            tier = "basic"
            display_name = "GROK-5"

        自定义模型会覆盖同 model_id 的内置模型。
        """
        from app.core.config import get_config

        custom_models_config = get_config("models", [])
        if not custom_models_config:
            logger.debug("No custom models in config, using built-in models only.")
            # 确保 _map 基于内置模型重建（处理 reload 场景）
            cls._map = {m.model_id: m for m in cls.BUILTIN_MODELS}
            return

        if not isinstance(custom_models_config, list):
            logger.warning(
                "Config 'models' should be an array of tables ([[models]]), skipping."
            )
            cls._map = {m.model_id: m for m in cls.BUILTIN_MODELS}
            return

        # 以内置模型为基础
        merged: Dict[str, ModelInfo] = {m.model_id: m for m in cls.BUILTIN_MODELS}
        loaded_count = 0
        overridden_count = 0

        for idx, model_cfg in enumerate(custom_models_config):
            if not isinstance(model_cfg, dict):
                logger.warning(f"Custom model #{idx}: expected dict, got {type(model_cfg).__name__}, skipping.")
                continue

            # 必填字段检查
            model_id = model_cfg.get("model_id")
            if not model_id:
                logger.warning(f"Custom model #{idx}: missing 'model_id', skipping.")
                continue

            try:
                model_info = ModelInfo(
                    model_id=model_cfg["model_id"],
                    grok_model=model_cfg.get("grok_model", model_cfg["model_id"]),
                    model_mode=model_cfg.get("model_mode", "MODEL_MODE_FAST"),
                    tier=Tier(model_cfg.get("tier", "basic")),
                    cost=Cost(model_cfg.get("cost", "low")),
                    display_name=model_cfg.get("display_name", model_cfg["model_id"].upper()),
                    description=model_cfg.get("description", ""),
                    is_video=model_cfg.get("is_video", False),
                    is_image=model_cfg.get("is_image", False),
                )

                is_override = model_id in merged
                merged[model_id] = model_info
                loaded_count += 1
                if is_override:
                    overridden_count += 1
                    logger.debug(f"Custom model '{model_id}' overrides built-in model.")
                else:
                    logger.debug(f"Custom model '{model_id}' registered.")

            except Exception as e:
                logger.warning(f"Custom model #{idx} ('{model_id}'): invalid config - {e}, skipping.")
                continue

        cls._map = merged
        logger.info(
            f"Model registry loaded: {len(cls.BUILTIN_MODELS)} built-in + "
            f"{loaded_count} custom ({overridden_count} overrides) = "
            f"{len(cls._map)} total models."
        )

    @classmethod
    def reload(cls) -> None:
        """重新加载模型配置（运行时热重载）"""
        logger.info("Reloading model registry...")
        cls.load_from_config()

    @classmethod
    def get(cls, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return cls._map.get(model_id)

    @classmethod
    def list(cls) -> list[ModelInfo]:
        """获取所有模型"""
        return list(cls._map.values())

    @classmethod
    def valid(cls, model_id: str) -> bool:
        """模型是否有效"""
        return model_id in cls._map

    @classmethod
    def to_grok(cls, model_id: str) -> Tuple[str, str]:
        """转换为 Grok 参数"""
        model = cls.get(model_id)
        if not model:
            raise ValidationException(f"Invalid model ID: {model_id}")
        return model.grok_model, model.model_mode

    @classmethod
    def pool_for_model(cls, model_id: str) -> str:
        """根据模型选择 Token 池"""
        model = cls.get(model_id)
        if model and model.tier == Tier.SUPER:
            return "ssoSuper"
        return "ssoBasic"

    @classmethod
    def pool_candidates_for_model(cls, model_id: str) -> List[str]:
        """按优先级返回可用 Token 池列表"""
        model = cls.get(model_id)
        if model and model.tier == Tier.SUPER:
            return ["ssoSuper"]
        # 基础模型优先使用 basic 池，缺失时可回退到 super 池
        return ["ssoBasic", "ssoSuper"]


__all__ = ["ModelService"]
