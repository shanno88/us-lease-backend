"""
Database models for Paddle Billing Integration
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class TransactionStatus(str, Enum):
    """Transaction status enumeration"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class TransactionModel(BaseModel):
    """
    Transaction model for Paddle payments

    This represents a transaction record in the database.
    Note: For production, you should use SQLAlchemy or similar ORM.
    """

    # Primary fields
    id: str = Field(..., description="Primary key (UUID)")
    paddle_transaction_id: str = Field(..., description="Paddle transaction ID")
    user_id: str = Field(..., description="User ID from frontend session")

    # Product/Price info
    product_id: str = Field(..., description="Paddle product ID")
    price_id: str = Field(..., description="Paddle price ID")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code (e.g., USD)")

    # Status tracking
    status: TransactionStatus = Field(
        default=TransactionStatus.PENDING, description="Transaction status"
    )

    # Customer info
    customer_email: Optional[str] = Field(None, description="Customer email address")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Metadata
    metadata: Optional[dict] = Field(
        default_factory=dict, description="Additional metadata as JSON"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "paddle_transaction_id": "txn_01kgrgfw904t11mrz86vks7k7t",
                "user_id": "session_abc123",
                "product_id": "pro_01kgrhkyabt3244vn6hqgj3ype",
                "price_id": "pri_01kgrhp2wrthebpgwmn8eh5ssy",
                "amount": 9.90,
                "currency": "USD",
                "status": "completed",
                "customer_email": "user@example.com",
                "created_at": "2026-02-06T12:00:00Z",
                "updated_at": "2026-02-06T12:00:00Z",
                "metadata": {},
            }
        }


# In-memory storage for demo purposes
# In production, replace this with a proper database (PostgreSQL, MySQL, etc.)
TRANSACTIONS_STORE: dict = {
    # "transaction_id": TransactionModel instance
}


class CreateCheckoutRequest(BaseModel):
    """Request model for creating checkout session"""

    user_id: str = Field(..., description="User ID from frontend session")
    email: Optional[str] = Field(None, description="Customer email (optional)")


class CreateCheckoutResponse(BaseModel):
    """Response model for checkout creation"""

    success: bool
    checkout_url: Optional[str] = None
    transaction_id: Optional[str] = None
    error: Optional[str] = None


class WebhookEvent(BaseModel):
    """Paddle webhook event model"""

    event_id: str
    event_type: str
    data: dict
    occurred_at: datetime


class TransactionResponse(BaseModel):
    """Response model for transaction query"""

    success: bool
    transaction: Optional[TransactionModel] = None
    error: Optional[str] = None


class OrderHistoryResponse(BaseModel):
    """Response model for order history"""

    success: bool
    orders: list
    total_count: int
    error: Optional[str] = None
