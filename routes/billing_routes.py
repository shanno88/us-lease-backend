from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import logging
from typing import Optional, Dict, Any
import os
from datetime import datetime, timedelta
import json
import asyncpg

from services.paddle import (
    create_checkout,
    verify_webhook_signature,
    parse_webhook_event,
    PAYMENT_SUCCESS_EVENTS,
)
from store import ANALYSIS_STORE

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

DATABASE_URL = os.getenv("DATABASE_URL")


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


async def get_db_connection() -> asyncpg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return await asyncpg.connect(DATABASE_URL)


async def get_active_subscription(user_id: str) -> Optional[asyncpg.Record]:
    conn = await get_db_connection()
    try:
        row = await conn.fetchrow(
            '''
            SELECT "paddlePriceId", "currentPeriodEnd"
            FROM subscriptions
            WHERE "userId" = $1
              AND "currentPeriodEnd" > NOW()
            ''',
            user_id,
        )
        return row
    finally:
        await conn.close()


def get_analyses_count_for_user(user_id: str) -> int:
    count = 0
    for analysis in ANALYSIS_STORE.values():
        if isinstance(analysis, dict) and analysis.get("user_id") == user_id:
            count += 1
    return count


async def grant_user_access(
    user_id: str,
    plan: str,
    customer_email: Optional[str] = None,
    transaction_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    price_id: Optional[str] = None,
    paddle_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now()
    duration_days = get_subscription_duration(plan)
    expires_at_dt = now + timedelta(days=duration_days)

    conn = await get_db_connection()
    try:
        email_value = customer_email or user_id

        await conn.execute(
            '''
            INSERT INTO "user" (id, email, name)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            ''',
            user_id,
            email_value,
            None,
        )

        await conn.execute(
            '''
            INSERT INTO subscriptions ("userId", "paddleSubscriptionId", "paddleCustomerId", "paddlePriceId", "currentPeriodEnd")
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT ("userId") DO UPDATE
            SET "paddleSubscriptionId" = EXCLUDED."paddleSubscriptionId",
                "paddleCustomerId" = EXCLUDED."paddleCustomerId",
                "paddlePriceId" = EXCLUDED."paddlePriceId",
                "currentPeriodEnd" = EXCLUDED."currentPeriodEnd"
            ''',
            user_id,
            subscription_id,
            paddle_customer_id,
            price_id,
            expires_at_dt,
        )
    finally:
        await conn.close()

    expires_at = expires_at_dt.isoformat()

    logger.info(
        f"[ACCESS GRANTED] user_id={user_id}, plan={plan}, expires_at={expires_at}"
    )
    return {
        "is_paid": True,
        "plan": plan,
        "paid_at": now.isoformat(),
        "expires_at": expires_at,
        "customer_email": customer_email,
        "transaction_id": transaction_id,
        "subscription_id": subscription_id,
        "analysis_ids": [],
    }


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

    access = await grant_user_access(
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

        # Check if user already has valid access in subscriptions table
        active_subscription = await get_active_subscription(request.user_id)
        if active_subscription:
            logger.info(
                f"User {request.user_id} already has valid access via active subscription"
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
        paddle_customer_id = None
        customer_obj = data.get("customer", {})
        if customer_obj:
            customer_email = customer_obj.get("email")
            paddle_customer_id = customer_obj.get("id")
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

            await grant_user_access(
                user_id=user_id,
                plan=plan,
                customer_email=customer_email,
                transaction_id=transaction_id,
                subscription_id=subscription_id,
                price_id=price_id,
                paddle_customer_id=paddle_customer_id,
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
        row = await get_active_subscription(user_id)

        if not row:
            return {
                "success": True,
                "has_access": False,
                "is_paid": False,
                "plan": None,
                "expires_at": None,
                "days_remaining": 0,
                "analyses_count": get_analyses_count_for_user(user_id),
            }

        price_id = row.get("paddlePriceId")
        current_period_end = row.get("currentPeriodEnd")
        plan = get_plan_from_price_id(price_id) if price_id else None

        now = datetime.now()

        if current_period_end and current_period_end > now:
            days_remaining = (current_period_end - now).days
            return {
                "success": True,
                "has_access": True,
                "is_paid": True,
                "plan": plan or "yearly",
                "expires_at": current_period_end.isoformat(),
                "days_remaining": days_remaining,
                "analyses_count": get_analyses_count_for_user(user_id),
            }

        return {
            "success": True,
            "has_access": False,
            "is_paid": False,
            "plan": plan,
            "expires_at": current_period_end.isoformat() if current_period_end else None,
            "days_remaining": 0,
            "analyses_count": get_analyses_count_for_user(user_id),
        }

    except Exception as e:
        logger.exception(f"Error checking access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check access: {str(e)}")


@router.get("/check-payment-status/{analysis_id}")
async def check_payment_status(analysis_id: str):
    """Legacy endpoint for backward compatibility"""
    try:
        if analysis_id not in ANALYSIS_STORE:
            raise HTTPException(status_code=404, detail="Analysis not found")

        user_id = ANALYSIS_STORE[analysis_id].get("user_id")
        if not user_id:
            return {
                "success": True,
                "analysis_id": analysis_id,
                "paid": False,
            }

        row = await get_active_subscription(user_id)

        is_paid = bool(row)

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
