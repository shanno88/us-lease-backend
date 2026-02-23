from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    Text,
    Integer,
)
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    EXPIRED = "expired"
    PAUSED = "paused"


class LeaseStatus(str, Enum):
    PENDING = "pending"
    ANALYZED = "analyzed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    paddle_customer_id = Column(String(255), unique=True, nullable=True, index=True)
    name = Column(String(255), nullable=True)
    free_analysis_used = Column(Boolean, default=False)
    has_active_plan = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    subscriptions = relationship(
        "Subscription", back_populates="user", cascade="all, delete-orphan"
    )
    leases = relationship("Lease", back_populates="user", cascade="all, delete-orphan")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String(36), primary_key=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paddle_subscription_id = Column(String(255), unique=True, nullable=True, index=True)
    paddle_transaction_id = Column(String(255), nullable=True)
    product_id = Column(String(255), nullable=True)
    price_id = Column(String(255), nullable=True)
    status = Column(String(50), default=SubscriptionStatus.ACTIVE.value)
    amount = Column(Float, nullable=True)
    currency = Column(String(10), default="USD")
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False)
    canceled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="subscriptions")


class Lease(Base):
    __tablename__ = "leases"

    id = Column(String(36), primary_key=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(500), nullable=True)
    status = Column(String(50), default=LeaseStatus.PENDING.value)
    raw_text = Column(Text, nullable=True)
    analysis_result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="leases")
