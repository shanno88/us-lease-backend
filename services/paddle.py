import httpx
import logging
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CheckoutRequest(BaseModel):
    analysis_id: str


class CheckoutResponse(BaseModel):
    checkout_url: str
    transaction_id: str


class PaddleConfig:
    def __init__(self):
        self.vendor_id = None
        self.api_key = None
        self.client_token = None
        self.environment = "sandbox"
        self.price_id = None

    @classmethod
    def from_env(cls):
        import os

        config = cls()
        config.vendor_id = os.getenv("PADDLE_VENDOR_ID")
        config.api_key = os.getenv("PADDLE_API_KEY")
        config.client_token = os.getenv("PADDLE_CLIENT_TOKEN")
        config.environment = os.getenv("PADDLE_ENV", "sandbox")
        config.price_id = os.getenv("PADDLE_PRICE_ID")
        return config


def get_paddle_config() -> PaddleConfig:
    return PaddleConfig.from_env()


async def create_checkout(
    analysis_id: str, config: Optional[PaddleConfig] = None
) -> Dict[str, Any]:
    if config is None:
        config = get_paddle_config()

    if not all([config.price_id, config.vendor_id, config.api_key]):
        raise ValueError(
            "Paddle configuration is incomplete. Please set PADDLE_VENDOR_ID, PADDLE_API_KEY, and PADDLE_PRICE_ID in environment variables."
        )

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "items": [
            {
                "price_id": config.price_id,
                "quantity": 1,
            }
        ],
        "customer_email": None,
        "custom_data": {
            "analysis_id": analysis_id,
        },
        "settings": {
            "display_name": "Lease Analysis Full Report",
            "success_url": f"{os.getenv('FRONTEND_URL', 'http://localhost:5173')}/#/billing/success?analysis_id={analysis_id}",
            "cancel_url": f"{os.getenv('FRONTEND_URL', 'http://localhost:5173')}/#/pricing",
        },
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.paddle.com/transactions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            logger.info(
                f"Created checkout for analysis_id: {analysis_id}, transaction_id: {data.get('data', {}).get('id')}"
            )

            return {
                "checkout_url": data["data"]["checkout_url"],
                "transaction_id": data["data"]["id"],
            }
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Paddle API error: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(f"Failed to create checkout: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error creating checkout: {str(e)}")
            raise Exception(f"Failed to create checkout: {str(e)}")


async def verify_webhook_signature(
    payload: bytes, signature: str, config: Optional[PaddleConfig] = None
) -> bool:
    if config is None:
        config = get_paddle_config()

    if not config.api_key:
        logger.warning(
            "Paddle API key not configured, skipping webhook signature verification"
        )
        return True

    import hmac
    import hashlib

    received_hmac = hmac.new(
        config.api_key.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(received_hmac, signature)


def parse_webhook_event(
    event_data: Dict[str, Any],
) -> "tuple[str, Optional[str], Optional[str], Optional[str]]":
    event_type = event_data.get("event_type", "")
    data = event_data.get("data", {})
    transaction_id = data.get("id")
    custom_data = data.get("custom_data", {})
    analysis_id = custom_data.get("analysis_id") if custom_data else None
    user_id = custom_data.get("user_id") if custom_data else None

    return event_type, analysis_id, transaction_id, user_id


PAYMENT_SUCCESS_EVENTS = [
    "transaction.completed",
    "transaction.billed",
    "transaction.paid",
    "subscription.activated",
    "subscription.created",
    "subscription.renewed",
    "subscription.updated",
]
