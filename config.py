from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    PORT: int = 3001
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_PATH: str = "./qiyoga.db"

    # CORS
    ALLOWED_ORIGINS: str = (
        "https://qiyoga.xyz,http://localhost:5173,http://localhost:3000"
    )
    BACKEND_CORS_ORIGINS: str = (
        "http://localhost:5173,https://qiyoga.xyz,http://localhost:3000"
    )

    # Paddle / 支付
    PADDLE_VENDOR_ID: str = ""
    PADDLE_API_KEY: str = ""
    PADDLE_WEBHOOK_SECRET: str = ""

    # DeepSeek（老版本用这个）
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # Qwen OCR via DashScope (Alibaba Cloud)
    DASHSCOPE_API_KEY: str = ""
    QWEN_VL_MODEL: str = "qwen-vl-ocr"

    # DeepInfra（现在保留做实验用）
    DEEPINFRA_API_KEY: str = ""
    DEEPINFRA_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Test mode
    TEST_USER_BYPASS: bool = False
    TEST_USER_IDS: str = ""

    def should_bypass_test_user(self, user_id: str) -> bool:
        if not self.TEST_USER_BYPASS:
            return False
        if user_id.startswith("test_user"):
            return True
        allowed_ids = [
            uid.strip() for uid in self.TEST_USER_IDS.split(",") if uid.strip()
        ]
        return user_id in allowed_ids


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
