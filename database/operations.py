"""
Database operations for transactions
In production, replace this with proper ORM (SQLAlchemy, etc.)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from models.transaction import (
    TransactionModel,
    TransactionStatus,
    TRANSACTIONS_STORE,
)


class DatabaseOperations:
    """Database operations handler"""

    @staticmethod
    def create_transaction(
        paddle_transaction_id: str,
        user_id: str,
        product_id: str,
        price_id: str,
        amount: float,
        currency: str = "USD",
        customer_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionModel:
        """
        Create a new transaction record

        Args:
            paddle_transaction_id: Paddle's transaction ID
            user_id: User ID from frontend
            product_id: Paddle product ID
            price_id: Paddle price ID
            amount: Transaction amount
            currency: Currency code (default: USD)
            customer_email: Optional customer email
            metadata: Optional metadata dictionary

        Returns:
            TransactionModel instance
        """
        import uuid

        transaction = TransactionModel(
            id=str(uuid.uuid4()),
            paddle_transaction_id=paddle_transaction_id,
            user_id=user_id,
            product_id=product_id,
            price_id=price_id,
            amount=amount,
            currency=currency,
            status=TransactionStatus.PENDING,
            customer_email=customer_email,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata or {},
        )

        # Store in memory (in production, insert into database)
        TRANSACTIONS_STORE[paddle_transaction_id] = transaction

        return transaction

    @staticmethod
    def update_transaction_status(
        paddle_transaction_id: str,
        status: TransactionStatus,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[TransactionModel]:
        """
        Update transaction status

        Args:
            paddle_transaction_id: Paddle transaction ID
            status: New transaction status
            metadata_updates: Optional metadata to update

        Returns:
            Updated TransactionModel or None if not found
        """
        if paddle_transaction_id not in TRANSACTIONS_STORE:
            return None

        transaction = TRANSACTIONS_STORE[paddle_transaction_id]
        transaction.status = status
        transaction.updated_at = datetime.utcnow()

        if metadata_updates:
            transaction.metadata.update(metadata_updates)

        return transaction

    @staticmethod
    def get_transaction(paddle_transaction_id: str) -> Optional[TransactionModel]:
        """
        Get transaction by Paddle transaction ID

        Args:
            paddle_transaction_id: Paddle transaction ID

        Returns:
            TransactionModel or None if not found
        """
        return TRANSACTIONS_STORE.get(paddle_transaction_id)

    @staticmethod
    def get_transaction_by_id(transaction_id: str) -> Optional[TransactionModel]:
        """
        Get transaction by internal ID

        Args:
            transaction_id: Internal transaction ID

        Returns:
            TransactionModel or None if not found
        """
        for transaction in TRANSACTIONS_STORE.values():
            if transaction.id == transaction_id:
                return transaction
        return None

    @staticmethod
    def get_user_transactions(user_id: str) -> List[TransactionModel]:
        """
        Get all transactions for a user

        Args:
            user_id: User ID

        Returns:
            List of TransactionModel instances
        """
        return [txn for txn in TRANSACTIONS_STORE.values() if txn.user_id == user_id]

    @staticmethod
    def get_recent_transactions(
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[TransactionModel]:
        """
        Get recent transactions

        Args:
            limit: Maximum number of transactions to return
            user_id: Optional user ID filter

        Returns:
            List of TransactionModel instances, sorted by created_at descending
        """
        transactions = list(TRANSACTIONS_STORE.values())

        if user_id:
            transactions = [t for t in transactions if t.user_id == user_id]

        # Sort by created_at descending
        transactions.sort(key=lambda x: x.created_at, reverse=True)

        return transactions[:limit]


# Helper functions for user access management
class UserAccessManager:
    """Manages user 30-day access passes"""

    @staticmethod
    def grant_access(user_id: str, expires_in_days: int = 30) -> Dict[str, Any]:
        """
        Grant 30-day access to a user

        Args:
            user_id: User ID
            expires_in_days: Number of days until expiration (default: 30)

        Returns:
            Dictionary with access details
        """
        from routes.lease_routes import USER_ACCESS_STORE
        from datetime import timedelta

        now = datetime.utcnow()
        expires_at = now + timedelta(days=expires_in_days)

        USER_ACCESS_STORE[user_id] = {
            "paid_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "analysis_ids": USER_ACCESS_STORE.get(user_id, {}).get("analysis_ids", []),
        }

        return {
            "user_id": user_id,
            "granted_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "days_until_expiration": expires_in_days,
        }

    @staticmethod
    def get_access_status(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user access status

        Args:
            user_id: User ID

        Returns:
            Dictionary with access status or None if user not found
        """
        from routes.lease_routes import USER_ACCESS_STORE

        if user_id not in USER_ACCESS_STORE:
            return {"has_access": False, "message": "User not found"}

        access = USER_ACCESS_STORE[user_id]
        now = datetime.utcnow()
        expires_at = datetime.fromisoformat(access["expires_at"])

        has_access = now < expires_at
        days_remaining = (expires_at - now).days if has_access else 0

        return {
            "has_access": has_access,
            "expires_at": access["expires_at"],
            "days_remaining": days_remaining,
            "analyses_count": len(access.get("analysis_ids", [])),
        }

    @staticmethod
    def revoke_access(user_id: str) -> bool:
        """
        Revoke user access

        Args:
            user_id: User ID

        Returns:
            True if successfully revoked
        """
        from routes.lease_routes import USER_ACCESS_STORE

        if user_id in USER_ACCESS_STORE:
            del USER_ACCESS_STORE[user_id]
            return True

        return False
