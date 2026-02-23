"""
Billing routes for Paddle integration
Handles checkout creation, webhooks, transaction status, and order history
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from services.paddle_billing import (
    PaddleClient,
    CheckoutRequest as PaddleCheckoutRequest,
    PAYMENT_SUCCESS_EVENTS,
    PAYMENT_FAILED_EVENTS,
    get_paddle_client,
)
from database.operations import DatabaseOperations, UserAccessManager
from models.transaction import TransactionModel, TransactionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])


class CreateCheckoutRequest(BaseModel):
    """Request model for creating checkout"""

    user_id: str
    email: Optional[str] = None


class CheckoutResponse(BaseModel):
    """Response model for checkout creation"""

    success: bool
    checkout_url: Optional[str] = None
    transaction_id: Optional[str] = None
    error: Optional[str] = None


class TransactionResponse(BaseModel):
    """Response model for transaction query"""

    success: bool
    transaction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OrderHistoryResponse(BaseModel):
    """Response model for order history"""

    success: bool
    orders: list
    total_count: int
    error: Optional[str] = None


@router.post("/checkout/create", response_model=CheckoutResponse)
async def create_checkout(request: CreateCheckoutRequest):
    """
    Create a Paddle checkout session

    Creates a checkout session and returns the checkout URL for the frontend to redirect to.
    """
    try:
        logger.info(f"Creating checkout for user_id: {request.user_id}")

        # Initialize Paddle client
        paddle_client = get_paddle_client()

        # Check if user already has active access
        access_status = UserAccessManager.get_access_status(request.user_id)
        if access_status and access_status.get("has_access"):
            logger.info(f"User {request.user_id} already has active access")
            return CheckoutResponse(
                success=True,
                checkout_url=None,
                transaction_id=None,
                error=None,
            )

        # Create checkout session
        result = await paddle_client.create_checkout_session(
            user_id=request.user_id,
            customer_email=request.email,
        )

        if not result["success"]:
            logger.error(f"Failed to create checkout: {result.get('error')}")
            return CheckoutResponse(
                success=False,
                checkout_url=None,
                transaction_id=None,
                error=result.get("error", "Failed to create checkout session"),
            )

        # Create transaction record with PENDING status
        transaction = DatabaseOperations.create_transaction(
            paddle_transaction_id=result["transaction_id"],
            user_id=request.user_id,
            product_id=paddle_client.config.product_id,
            price_id=paddle_client.config.price_id,
            amount=9.90,  # Fixed price
            currency="USD",
            customer_email=request.email,
        )

        logger.info(f"Checkout created successfully: {result['transaction_id']}")

        return CheckoutResponse(
            success=True,
            checkout_url=result["checkout_url"],
            transaction_id=result["transaction_id"],
            error=None,
        )

    except Exception as e:
        logger.exception(f"Error in create_checkout: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout: {str(e)}",
        )


@router.post("/webhook")
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Handle Paddle webhook events

    Receives and processes webhook events from Paddle for payment status updates.
    Verifies webhook signature for security.
    """
    try:
        # Get raw body for signature verification
        body = await request.body()

        # Get signature from headers
        signature = request.headers.get("paddle_signature", "")
        if not signature:
            logger.warning("Webhook received without signature")
            raise HTTPException(status_code=401, detail="Missing signature")

        # Initialize Paddle client
        paddle_client = get_paddle_client()

        # Verify webhook signature
        if not paddle_client.verify_webhook_signature(body, signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse event data
        event_data = await request.json()
        event_type, transaction_id, user_id = paddle_client.parse_webhook_event(
            event_data
        )

        logger.info(
            f"Webhook received: event_type={event_type}, "
            f"transaction_id={transaction_id}, user_id={user_id}"
        )

        # Process payment success events
        if event_type in PAYMENT_SUCCESS_EVENTS:
            if transaction_id and user_id:
                # Update transaction status to COMPLETED
                transaction = DatabaseOperations.update_transaction_status(
                    paddle_transaction_id=transaction_id,
                    status=TransactionStatus.COMPLETED,
                    metadata_updates={
                        "webhook_received_at": datetime.utcnow().isoformat(),
                        "event_type": event_type,
                    },
                )

                if transaction:
                    logger.info(f"Transaction {transaction_id} marked as completed")

                # Grant 30-day access to user
                UserAccessManager.grant_access(user_id=user_id, expires_in_days=30)

                logger.info(f"Granted 30-day access to user {user_id}")
            else:
                logger.warning(
                    f"Payment success event missing transaction_id or user_id"
                )

        # Process payment failed events
        elif event_type in PAYMENT_FAILED_EVENTS:
            if transaction_id:
                # Update transaction status to FAILED
                DatabaseOperations.update_transaction_status(
                    paddle_transaction_id=transaction_id,
                    status=TransactionStatus.FAILED,
                    metadata_updates={
                        "failure_reason": "Payment failed",
                        "event_type": event_type,
                    },
                )

                logger.info(f"Transaction {transaction_id} marked as failed")
            else:
                logger.warning(f"Payment failed event missing transaction_id")

        # Acknowledge webhook
        return {"status": "success"}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Webhook processing failed: {str(e)}",
        )


@router.get("/transaction/{transaction_id}", response_model=TransactionResponse)
async def get_transaction_status(transaction_id: str):
    """
    Query transaction status by transaction ID

    Returns the details and status of a specific transaction.
    """
    try:
        logger.info(f"Querying transaction: {transaction_id}")

        # Try to get by Paddle transaction ID first
        transaction = DatabaseOperations.get_transaction(transaction_id)

        # If not found, try internal ID
        if not transaction:
            transaction = DatabaseOperations.get_transaction_by_id(transaction_id)

        if not transaction:
            logger.warning(f"Transaction not found: {transaction_id}")
            return TransactionResponse(
                success=False,
                transaction=None,
                error="Transaction not found",
            )

        # Convert to dict for response
        transaction_dict = transaction.model_dump()

        logger.info(f"Returning transaction status for {transaction_id}")

        return TransactionResponse(
            success=True,
            transaction=transaction_dict,
            error=None,
        )

    except Exception as e:
        logger.exception(f"Error querying transaction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query transaction: {str(e)}",
        )


@router.get("/orders/{user_id}", response_model=OrderHistoryResponse)
async def get_user_orders(user_id: str, limit: Optional[int] = 10):
    """
    Get user's order history

    Returns all transactions for a specific user, sorted by most recent.
    """
    try:
        logger.info(f"Fetching orders for user: {user_id}")

        # Get transactions for user
        transactions = DatabaseOperations.get_user_transactions(user_id)

        # Sort by created_at descending
        transactions.sort(key=lambda x: x.created_at, reverse=True)

        # Limit results if specified
        if limit and limit < len(transactions):
            transactions = transactions[:limit]

        # Convert to dicts
        orders = [txn.model_dump() for txn in transactions]

        logger.info(f"Returning {len(orders)} orders for user {user_id}")

        return OrderHistoryResponse(
            success=True,
            orders=orders,
            total_count=len(transactions),
            error=None,
        )

    except Exception as e:
        logger.exception(f"Error fetching orders: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch orders: {str(e)}",
        )


@router.get("/check-access/{user_id}")
async def check_user_access(user_id: str):
    """
    Check user's 30-day access status

    Returns whether the user has active access and expiration details.
    """
    try:
        logger.info(f"Checking access for user: {user_id}")

        access_status = UserAccessManager.get_access_status(user_id)

        if not access_status:
            return {
                "success": True,
                "has_access": False,
                "message": "User not found",
            }

        return {
            "success": True,
            "has_access": access_status.get("has_access", False),
            "expires_at": access_status.get("expires_at"),
            "days_remaining": access_status.get("days_remaining", 0),
            "analyses_count": access_status.get("analyses_count", 0),
        }

    except Exception as e:
        logger.exception(f"Error checking access: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check access: {str(e)}",
        )
