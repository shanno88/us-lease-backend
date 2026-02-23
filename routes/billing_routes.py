from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import logging
from typing import Optional, Dict, Any
import os
from datetime import datetime, timedelta
import json

from services.paddle import (
    create_checkout,
    verify_webhook_signature,
    parse_webhook_event,
    PAYMENT_SUCCESS_EVENTS,
)
from routes.lease_routes import USER_ACCESS_STORE

logger = logging.getLogger(__name__)

router = APIRouter(tags=["billing"])

PENDING_PAYMENTS: Dict[str, Dict[str, Any]] = {}

SANDBOX_PRICE_IDS = {
    "monthly": "pri_01khstd1ehd0v9xs84ev0wttg6",
    "yearly": "pri_01khstexva93m2jzdw38cx0gj8",
}

LIVE_PRICE_IDS = {
    "monthly": "pri_LIVE_MONTHLY_PLACEHOLDER",
    "yearly": "pri_LIVE_YEARLY_PLACEHOLDER",
}


def get_plan_from_price_id(price_id: str) -> str:
    for plan, pid in SANDBOX_PRICE_IDS.items():
        if pid == price_id:
            return plan
    for plan, pid in LIVE_PRICE_IDS.items():
        if pid == price_id:
            return plan
    return "yearly"


def get_subscription_duration(plan: str) -> int:
    if plan == "monthly":
        return 30
    return 365


def grant_user_access(
    user_id: str,
    plan: str,
    customer_email: Optional[str] = None,
    transaction_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now()
    duration_days = get_subscription_duration(plan)
    expires_at = (now + timedelta(days=duration_days)).isoformat()

    existing_analysis_ids = []
    if user_id in USER_ACCESS_STORE:
        existing_analysis_ids = USER_ACCESS_STORE[user_id].get("analysis_ids", [])

    USER_ACCESS_STORE[user_id] = {
        "is_paid": True,
        "plan": plan,
        "paid_at": now.isoformat(),
        "expires_at": expires_at,
        "customer_email": customer_email,
        "transaction_id": transaction_id,
        "subscription_id": subscription_id,
        "analysis_ids": existing_analysis_ids,
    }

    logger.info(
        f"[ACCESS GRANTED] user_id={user_id}, plan={plan}, expires_at={expires_at}"
    )
    return USER_ACCESS_STORE[user_id]


class CreateCheckoutRequest(BaseModel):
    user_id: str


class CheckoutResponse(BaseModel):
    success: bool
    checkout_url: Optional[str] = None
    transaction_id: Optional[str] = None
    error: Optional[str] = None


class GrantAccessRequest(BaseModel):
    user_id: str
    transaction_id: Optional[str] = None
    customer_email: Optional[str] = None
    price_id: Optional[str] = None


class RegisterPendingRequest(BaseModel):
    user_id: str
    checkout_id: Optional[str] = None


@router.post("/register-pending")
async def register_pending_payment(request: RegisterPendingRequest):
    logger.info(
        f"[REGISTER PENDING] user_id={request.user_id}, checkout_id={request.checkout_id}"
    )

    key = request.checkout_id or request.user_id
    PENDING_PAYMENTS[key] = {
        "user_id": request.user_id,
        "checkout_id": request.checkout_id,
        "registered_at": datetime.now().isoformat(),
    }

    logger.info(
        f"[REGISTER PENDING] Stored mapping: {key} -> user_id={request.user_id}"
    )
    return {"success": True, "key": key}


@router.post("/grant-access")
async def grant_access_direct(request: GrantAccessRequest):
    logger.info(
        f"[GRANT ACCESS DIRECT] user_id={request.user_id}, transaction_id={request.transaction_id}"
    )

    plan = "yearly"
    if request.price_id:
        plan = get_plan_from_price_id(request.price_id)

    access = grant_user_access(
        user_id=request.user_id,
        plan=plan,
        customer_email=request.customer_email,
        transaction_id=request.transaction_id,
    )

    return {
        "success": True,
        "has_access": True,
        "plan": access["plan"],
        "expires_at": access["expires_at"],
    }


@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_checkout_session(request: CreateCheckoutRequest):
    try:
        logger.info(f"Creating checkout for user_id: {request.user_id}")

        # Check if user already has valid access
        now = datetime.now()
        if request.user_id in USER_ACCESS_STORE:
            access = USER_ACCESS_STORE[request.user_id]
            if "expires_at" in access:
                expires_at = datetime.fromisoformat(access["expires_at"])
                if now < expires_at:
                    logger.info(
                        f"User {request.user_id} already has valid access until {expires_at}"
                    )
                    return CheckoutResponse(
                        success=True,
                        checkout_url=None,
                        transaction_id=None,
                        error=None,
                    )

        # Create Paddle checkout with user_id in metadata
        result = await create_checkout_for_user(request.user_id)

        logger.info(f"Checkout created successfully: {result['transaction_id']}")

        return CheckoutResponse(
            success=True,
            checkout_url=result["checkout_url"],
            transaction_id=result["transaction_id"],
        )

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return CheckoutResponse(
            success=False, error=f"Payment system configuration error: {str(e)}"
        )

    except Exception as e:
        logger.exception(f"Error creating checkout: {str(e)}")
        return CheckoutResponse(
            success=False, error=f"Failed to create checkout: {str(e)}"
        )


async def create_checkout_for_user(user_id: str) -> dict:
    """Create a Paddle checkout session for the given user"""
    import httpx

    config = {
        "vendor_id": os.getenv("PADDLE_VENDOR_ID"),
        "api_key": os.getenv("PADDLE_API_KEY"),
        "client_token": os.getenv("PADDLE_CLIENT_TOKEN"),
        "environment": os.getenv("PADDLE_ENV", "sandbox"),
        "price_id": os.getenv("PADDLE_PRICE_ID"),
    }

    if not all([config["price_id"], config["vendor_id"], config["api_key"]]):
        raise ValueError(
            "Paddle configuration is incomplete. Please set PADDLE_VENDOR_ID, PADDLE_API_KEY, and PADDLE_PRICE_ID in environment variables."
        )

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "items": [
            {
                "price_id": config["price_id"],
                "quantity": 1,
            }
        ],
        "customer_email": None,
        "custom_data": {
            "user_id": user_id,
        },
        "settings": {
            "display_name": "30-Day Unlimited Lease Analysis Access",
            "success_url": f"{os.getenv('FRONTEND_URL', 'https://qiyoga.xyz')}/#/billing/success?user_id={user_id}",
            "cancel_url": f"{os.getenv('FRONTEND_URL', 'https://qiyoga.xyz')}/#/pricing",
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
                f"Created checkout for user_id: {user_id}, transaction_id: {data.get('data', {}).get('id')}"
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


@router.post("/webhook")
async def paddle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    try:
        body = await request.body()

        signature = request.headers.get("paddle_signature", "")
        if not signature:
            logger.warning("[WEBHOOK] Received without signature")
            raise HTTPException(status_code=401, detail="Missing signature")

        if not await verify_webhook_signature(body, signature):
            logger.warning("[WEBHOOK] Invalid signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

        event_data = await request.json()
        event_type = event_data.get("event_type", "")
        data = event_data.get("data", {})

        logger.info(f"[WEBHOOK] ========== START ==========")
        logger.info(f"[WEBHOOK] event_type: {event_type}")
        logger.info(f"[WEBHOOK] event_id: {event_data.get('event_id', 'N/A')}")
        logger.info(f"[WEBHOOK] Full data keys: {list(data.keys())}")

        custom_data = data.get("custom_data") or {}
        logger.info(f"[WEBHOOK] custom_data from data.custom_data: {custom_data}")
        user_id = custom_data.get("user_id") if custom_data else None
        logger.info(f"[WEBHOOK] user_id from custom_data: {user_id}")

        customer_email = None
        customer_obj = data.get("customer", {})
        if customer_obj:
            customer_email = customer_obj.get("email")
            logger.info(
                f"[WEBHOOK] customer_email from data.customer.email: {customer_email}"
            )

        checkout_custom_data = None
        checkout_obj = data.get("checkout", {})
        if checkout_obj:
            checkout_custom_data = checkout_obj.get("custom_data") or {}
            logger.info(f"[WEBHOOK] checkout.custom_data: {checkout_custom_data}")
            if not user_id and checkout_custom_data:
                user_id = checkout_custom_data.get("user_id")

        transaction_id = data.get("id")
        logger.info(f"[WEBHOOK] transaction_id (data.id): {transaction_id}")

        price_id = None
        subscription_id = None

        items = data.get("items", [])
        if items and len(items) > 0:
            first_item = items[0]
            price_obj = first_item.get("price", {})
            price_id = price_obj.get("id") if price_obj else None
            logger.info(f"[WEBHOOK] price_id from items[0].price.id: {price_id}")

        if "subscription" in data:
            subscription_id = data["subscription"].get("id")
            logger.info(
                f"[WEBHOOK] subscription_id from data.subscription.id: {subscription_id}"
            )

        if not user_id and transaction_id and transaction_id in PENDING_PAYMENTS:
            pending = PENDING_PAYMENTS[transaction_id]
            user_id = pending.get("user_id")
            logger.info(
                f"[WEBHOOK] user_id from PENDING_PAYMENTS[transaction_id]: {user_id}"
            )

        if not user_id and customer_email:
            logger.info(
                f"[WEBHOOK] No user_id found, using customer_email as user_id: {customer_email}"
            )
            user_id = customer_email

        logger.info(f"[WEBHOOK] FINAL user_id: {user_id}")
        logger.info(f"[WEBHOOK] FINAL price_id: {price_id}")
        logger.info(f"[WEBHOOK] FINAL customer_email: {customer_email}")

        if event_type in PAYMENT_SUCCESS_EVENTS and user_id:
            plan = "yearly"
            if price_id:
                plan = get_plan_from_price_id(price_id)
            logger.info(f"[WEBHOOK] Determined plan: {plan}")

            grant_user_access(
                user_id=user_id,
                plan=plan,
                customer_email=customer_email,
                transaction_id=transaction_id,
                subscription_id=subscription_id,
            )

            if transaction_id and transaction_id in PENDING_PAYMENTS:
                del PENDING_PAYMENTS[transaction_id]
                logger.info(
                    f"[WEBHOOK] Cleaned up pending payment for {transaction_id}"
                )

            logger.info(f"[WEBHOOK] ========== SUCCESS ==========")
        else:
            logger.warning(
                f"[WEBHOOK] NOT PROCESSED - event_type={event_type}, user_id={user_id}"
            )
            logger.warning(
                f"[WEBHOOK] PAYMENT_SUCCESS_EVENTS: {PAYMENT_SUCCESS_EVENTS}"
            )
            logger.warning(f"[WEBHOOK] ========== SKIPPED ==========")

        return {"status": "success"}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"[WEBHOOK] ERROR: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Webhook processing failed: {str(e)}"
        )


@router.get("/check-access")
async def check_user_access(
    user_id: str = Query(..., description="User ID from frontend session"),
):
    try:
        if user_id not in USER_ACCESS_STORE:
            return {
                "success": True,
                "has_access": False,
                "is_paid": False,
                "plan": None,
                "expires_at": None,
                "days_remaining": 0,
                "analyses_count": 0,
            }

        access = USER_ACCESS_STORE[user_id]
        now = datetime.now()

        if "expires_at" not in access:
            return {
                "success": True,
                "has_access": False,
                "is_paid": access.get("is_paid", False),
                "plan": access.get("plan"),
                "expires_at": None,
                "days_remaining": 0,
                "analyses_count": len(access.get("analysis_ids", [])),
            }

        expires_at = datetime.fromisoformat(access["expires_at"])

        if now < expires_at:
            days_remaining = (expires_at - now).days
            return {
                "success": True,
                "has_access": True,
                "is_paid": True,
                "plan": access.get("plan", "yearly"),
                "expires_at": access["expires_at"],
                "days_remaining": days_remaining,
                "analyses_count": len(access.get("analysis_ids", [])),
            }
        else:
            return {
                "success": True,
                "has_access": False,
                "is_paid": False,
                "plan": access.get("plan"),
                "expires_at": access["expires_at"],
                "days_remaining": 0,
                "analyses_count": len(access.get("analysis_ids", [])),
            }

    except Exception as e:
        logger.exception(f"Error checking access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check access: {str(e)}")


@router.get("/check-payment-status/{analysis_id}")
async def check_payment_status(analysis_id: str):
    """Legacy endpoint for backward compatibility"""
    try:
        from routes.lease_routes import ANALYSIS_STORE

        if analysis_id not in ANALYSIS_STORE:
            raise HTTPException(status_code=404, detail="Analysis not found")

        user_id = ANALYSIS_STORE[analysis_id].get("user_id")
        if not user_id:
            return {
                "success": True,
                "analysis_id": analysis_id,
                "paid": False,
            }

        if user_id not in USER_ACCESS_STORE:
            return {
                "success": True,
                "analysis_id": analysis_id,
                "paid": False,
            }

        access = USER_ACCESS_STORE[user_id]
        now = datetime.now()

        if "expires_at" in access:
            expires_at = datetime.fromisoformat(access["expires_at"])
            is_paid = now < expires_at
        else:
            is_paid = False

        return {
            "success": True,
            "analysis_id": analysis_id,
            "paid": is_paid,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error checking payment status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check payment status: {str(e)}"
        )
