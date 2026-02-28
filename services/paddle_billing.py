"""
Paddle Billing API Service
Latest Paddle Billing API integration
"""

import httpx
import logging
import os
import hmac
import hashlib
import json
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckoutRequest(BaseModel):
    """Request model for checkout creation"""

    user_id: str
    email: Optional[str] = None


class CheckoutResponse(BaseModel):
    """Response model for checkout creation"""

    success: bool
    checkout_url: Optional[str] = None
    transaction_id: Optional[str] = None
    error: Optional[str] = None


class PaddleConfig:
    """Paddle Billing API configuration"""

    def __init__(self):
        # API credentials
        self.api_key = os.getenv("PADDLE_API_KEY")

        # Product/Price IDs
        self.product_id = os.getenv(
            "PADDLE_PRODUCT_ID", "pro_01kgrhkyabt3244vn6hqgj3ype"
        )
        self.price_id = os.getenv("PADDLE_PRICE_ID", "pri_01kgrhp2wrthebpgwmn8eh5ssy")

        # Environment
        self.environment = os.getenv(
            "PADDLE_ENV", "production"
        )  # production or sandbox
        self.api_base_url = self._get_api_base_url()

        # Webhook configuration
        self.webhook_secret = os.getenv("PADDLE_WEBHOOK_SECRET")

        # Frontend URLs
        self.frontend_url = os.getenv("FRONTEND_URL", "https://qiyoga.xyz")

    def _get_api_base_url(self) -> str:
        """Get API base URL based on environment"""
        if self.environment == "production":
            return "https://api.paddle.com"
        elif self.environment == "sandbox":
            return "https://sandbox-api.paddle.com"
        else:
            raise ValueError(f"Unknown Paddle environment: {self.environment}")

    def is_configured(self) -> bool:
        """Check if Paddle is properly configured"""
        return bool(self.api_key and self.product_id and self.price_id)

    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls()


def get_paddle_config() -> PaddleConfig:
    """Get Paddle configuration singleton"""
    return PaddleConfig.from_env()


class PaddleClient:
    """Paddle Billing API client"""

    def __init__(self, config: Optional[PaddleConfig] = None):
        self.config = config or get_paddle_config()

        if not self.config.is_configured():
            raise ValueError(
                "Paddle configuration is incomplete. "
                "Please set PADDLE_API_KEY, PADDLE_PRODUCT_ID, and PADDLE_PRICE_ID in environment variables."
            )

        logger.info(
            f"Paddle client initialized (environment: {self.config.environment}, "
            f"product: {self.config.product_id}, price: {self.config.price_id})"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    async def create_checkout_session(
        self,
        user_id: str,
        customer_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a Paddle checkout session

        Args:
            user_id: User ID from frontend session
            customer_email: Optional customer email address

        Returns:
            Dictionary with checkout_url and transaction_id
        """
        headers = self._get_headers()

        # Build checkout payload according to Paddle Billing API
        payload = {
            "items": [
                {
                    "price_id": self.config.price_id,
                    "quantity": 1,
                }
            ],
            "customer_email": customer_email,
            "custom_data": {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "settings": {
                "display_name": "TutorBox - 30-Day Unlimited Access",
                "success_url": f"{self.config.frontend_url}/#/billing/success?user_id={user_id}",
                "cancel_url": f"{self.config.frontend_url}/#/pricing",
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Create checkout session using Paddle Billing API
                response = await client.post(
                    f"{self.config.api_base_url}/transactions",
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                data = response.json()

                checkout_url = data.get("data", {}).get("checkout_url")
                transaction_id = data.get("data", {}).get("id")

                if not checkout_url or not transaction_id:
                    raise Exception(
                        "Invalid response from Paddle API: missing checkout_url or transaction_id"
                    )

                logger.info(
                    f"Created checkout session: user_id={user_id}, "
                    f"transaction_id={transaction_id}"
                )

                return {
                    "success": True,
                    "checkout_url": checkout_url,
                    "transaction_id": transaction_id,
                }

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Paddle API error: {e.response.status_code} - {e.response.text}"
                )
                error_data = None
                try:
                    error_data = e.response.json()
                except:
                    pass

                error_msg = f"Paddle API error: {e.response.status_code}"
                if error_data:
                    error_msg = error_data.get("detail", error_msg)

                return {
                    "success": False,
                    "checkout_url": None,
                    "transaction_id": None,
                    "error": error_msg,
                }

            except httpx.TimeoutException:
                logger.error("Paddle API timeout")
                return {
                    "success": False,
                    "checkout_url": None,
                    "transaction_id": None,
                    "error": "Request to Paddle API timed out",
                }

            except Exception as e:
                logger.exception(f"Unexpected error creating checkout: {str(e)}")
                return {
                    "success": False,
                    "checkout_url": None,
                    "transaction_id": None,
                    "error": f"Failed to create checkout: {str(e)}",
                }

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Verify Paddle webhook signature

        Paddle Billing API uses HMAC-SHA256 signature for webhook verification.

        Args:
            payload: Raw request body (bytes)
            signature: Signature from request header
            timestamp: Optional timestamp from request header

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.config.api_key:
            logger.warning(
                "Paddle API key not configured, skipping webhook signature verification"
            )
            return True

        try:
            # Paddle uses HMAC-SHA256 for signature verification
            received_hmac = hmac.new(
                self.config.api_key.encode(),
                payload,
                hashlib.sha256,
            ).hexdigest()

            # Compare securely
            is_valid = hmac.compare_digest(received_hmac, signature)

            if not is_valid:
                logger.warning("Invalid webhook signature received")
            else:
                logger.info("Webhook signature verified successfully")

            return is_valid

        except Exception as e:
            logger.exception(f"Error verifying webhook signature: {str(e)}")
            return False

    def parse_webhook_event(
        self,
        event_data: Dict[str, Any],
    ):
        """
        Parse Paddle webhook event

        Args:
            event_data: Raw webhook event data

        Returns:
            event_type, transaction_id, user_id
        """
        event_type = event_data.get("event_type", "")

        # Extract data payload
        data = event_data.get("data", {})
        transaction_id = data.get("id")

        # Extract custom_data
        custom_data = data.get("custom_data", {})
        user_id = custom_data.get("user_id") if custom_data else None

        logger.info(
            f"Parsed webhook event: event_type={event_type}, "
            f"transaction_id={transaction_id}, user_id={user_id}"
        )

        return event_type, transaction_id, user_id


# Event types we care about
PAYMENT_SUCCESS_EVENTS = [
    "transaction.completed",
    "transaction.billed",
    "subscription.activated",
]

PAYMENT_FAILED_EVENTS = [
    "transaction.payment_failed",
    "transaction.failed",
]


def get_paddle_client() -> PaddleClient:
    """Get Paddle client singleton"""
    return PaddleClient()
