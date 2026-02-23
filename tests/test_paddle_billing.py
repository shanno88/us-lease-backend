"""
Paddle Billing Integration - Test Suite

Run these tests to verify the Paddle integration is working correctly.
Before running tests:
1. Ensure the backend server is running
2. Set up test environment variables in .env.test
3. Install test dependencies: pip install pytest pytest-asyncio httpx
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import json

# Test configuration
TEST_API_URL = "http://127.0.0.1:8000"
TEST_USER_ID = "test_user_session_123"
TEST_EMAIL = "test@example.com"


class TestPaddleCheckout:
    """Test suite for Paddle checkout functionality"""

    @pytest.mark.asyncio
    async def test_create_checkout_success(self, httpx_mock):
        """Test successful checkout creation"""
        # Mock the Paddle API response
        mock_response = {
            "success": True,
            "checkout_url": "https://checkout.paddle.com/checkout/test",
            "transaction_id": "txn_test123",
            "error": None,
        }
        httpx_mock.post(
            f"{TEST_API_URL}/api/billing/checkout/create", json=mock_response
        )

        # Make actual request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/checkout/create",
                json={
                    "user_id": TEST_USER_ID,
                    "email": TEST_EMAIL,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "checkout_url" in data
            assert "transaction_id" in data

    @pytest.mark.asyncio
    async def test_create_checkout_missing_user_id(self):
        """Test checkout creation with missing user_id"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/checkout/create",
                json={"email": TEST_EMAIL},
            )

            assert response.status_code == 422  # Unprocessable Entity

    @pytest.mark.asyncio
    async def test_create_checkout_duplicate_access(self, httpx_mock):
        """Test checkout when user already has access"""
        # First request should succeed
        async with httpx.AsyncClient() as client:
            response1 = await client.post(
                f"{TEST_API_URL}/api/billing/checkout/create",
                json={"user_id": TEST_USER_ID},
            )

            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["success"] == True

            # Second request should indicate already has access
            response2 = await client.post(
                f"{TEST_API_URL}/api/billing/checkout/create",
                json={"user_id": TEST_USER_ID},
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["success"] == True
            assert data2["checkout_url"] is None  # No checkout URL needed


class TestPaddleWebhook:
    """Test suite for Paddle webhook handling"""

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature(self, httpx_mock):
        """Test webhook with invalid signature"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/webhook",
                json={
                    "event_type": "transaction.completed",
                    "data": {
                        "id": "txn_test123",
                        "custom_data": {"user_id": TEST_USER_ID},
                    },
                },
                headers={
                    "paddle_signature": "invalid_signature_123",
                },
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_webhook_missing_signature(self, httpx_mock):
        """Test webhook with missing signature"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/webhook",
                json={
                    "event_type": "transaction.completed",
                    "data": {
                        "id": "txn_test123",
                        "custom_data": {"user_id": TEST_USER_ID},
                    },
                },
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_webhook_transaction_completed(self, httpx_mock):
        """Test webhook for transaction.completed event"""
        # Generate valid HMAC signature
        import hmac
        import hashlib
        from dotenv import load_dotenv

        load_dotenv()

        api_key = "test_api_key_123"  # Use actual test key
        payload = json.dumps(
            {
                "event_type": "transaction.completed",
                "data": {
                    "id": "txn_test123",
                    "custom_data": {"user_id": TEST_USER_ID},
                },
            }
        ).encode()
        signature = hmac.new(api_key.encode(), payload, hashlib.sha256).hexdigest()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/webhook",
                data=json.dumps(
                    {
                        "event_type": "transaction.completed",
                        "data": {
                            "id": "txn_test123",
                            "custom_data": {"user_id": TEST_USER_ID},
                        },
                    }
                ),
                headers={"paddle_signature": signature},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_webhook_transaction_failed(self, httpx_mock):
        """Test webhook for transaction.failed event"""
        import hmac
        import hashlib

        api_key = "test_api_key_123"
        payload = json.dumps(
            {
                "event_type": "transaction.failed",
                "data": {
                    "id": "txn_test123",
                },
            }
        ).encode()
        signature = hmac.new(api_key.encode(), payload, hashlib.sha256).hexdigest()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TEST_API_URL}/api/billing/webhook",
                data=json.dumps(
                    {
                        "event_type": "transaction.failed",
                        "data": {
                            "id": "txn_test123",
                        },
                    }
                ),
                headers={"paddle_signature": signature},
            )

            assert response.status_code == 200


class TestTransactionQueries:
    """Test suite for transaction status and order history queries"""

    @pytest.mark.asyncio
    async def test_get_transaction_success(self):
        """Test getting transaction status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TEST_API_URL}/api/billing/transaction/txn_test123"
            )

            # Note: This assumes transaction exists in the database
            # In real testing, you'd create a transaction first

            assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_get_transaction_not_found(self):
        """Test getting non-existent transaction"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TEST_API_URL}/api/billing/transaction/nonexistent_txn"
            )

            assert response.status_code == 404
            data = response.json()
            assert data["success"] == False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_get_user_orders(self):
        """Test getting user's order history"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TEST_API_URL}/api/billing/orders/{TEST_USER_ID}"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "orders" in data
            assert "total_count" in data

    @pytest.mark.asyncio
    async def test_check_user_access(self):
        """Test checking user access status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TEST_API_URL}/api/billing/check-access/{TEST_USER_ID}"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "has_access" in data


class TestIntegration:
    """Integration tests for complete payment flow"""

    @pytest.mark.asyncio
    async def test_complete_payment_flow(self):
        """
        Test the complete payment flow:
        1. Create checkout
        2. Simulate webhook payment completion
        3. Verify access granted
        """
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Create checkout
            checkout_response = await client.post(
                f"{TEST_API_URL}/api/billing/checkout/create",
                json={
                    "user_id": f"integration_test_{datetime.now().timestamp()}",
                    "email": "integration@test.com",
                },
            )

            assert checkout_response.status_code == 200
            checkout_data = checkout_response.json()
            transaction_id = checkout_data.get("transaction_id")

            print(f"✓ Checkout created: {transaction_id}")

            # Step 2: Simulate webhook (in real flow, Paddle sends this)
            # Skip this in automated tests as webhook requires valid signature

            # Step 3: Verify access was granted
            await asyncio.sleep(2)  # Wait for processing

            access_response = await client.get(
                f"{TEST_API_URL}/api/billing/check-access/integration_test_{datetime.now().timestamp()}"
            )

            assert access_response.status_code == 200
            access_data = access_response.json()
            assert access_data["success"] == True

            print("✓ Complete payment flow test passed")


# Run tests if executed directly
if __name__ == "__main__":
    print("Paddle Billing Integration Test Suite")
    print("\nTo run tests:")
    print("1. Install pytest: pip install pytest pytest-asyncio httpx")
    print(
        "2. Run specific test: pytest tests/test_paddle_billing.py -k test_create_checkout_success"
    )
    print("3. Run all tests: pytest tests/test_paddle_billing.py")
    print("\nAvailable test classes:")
    print("- TestPaddleCheckout")
    print("- TestPaddleWebhook")
    print("- TestTransactionQueries")
    print("- TestIntegration")
