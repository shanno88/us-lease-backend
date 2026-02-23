"""
Database Migration Script for Paddle Billing Integration

This script creates the necessary database tables for the billing system.
For production, you should use a proper migration tool like Alembic.
"""


def create_transactions_table_sql():
    """
    SQL to create transactions table for PostgreSQL
    """
    sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        id VARCHAR(255) PRIMARY KEY,
        paddle_transaction_id VARCHAR(255) UNIQUE NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        product_id VARCHAR(255) NOT NULL,
        price_id VARCHAR(255) NOT NULL,
        amount DECIMAL(10, 2) NOT NULL,
        currency VARCHAR(10) DEFAULT 'USD',
        status VARCHAR(50) NOT NULL DEFAULT 'pending',
        customer_email VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}'::jsonb
    );

    CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_paddle_transaction_id ON transactions(paddle_transaction_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
    CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);
    """
    return sql


def create_user_access_table_sql():
    """
    SQL to create user_access table for PostgreSQL
    This table tracks user 30-day access passes
    """
    sql = """
    CREATE TABLE IF NOT EXISTS user_access (
        user_id VARCHAR(255) PRIMARY KEY,
        paid_at TIMESTAMP WITH TIME ZONE NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        analysis_ids TEXT[] DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_user_access_user_id ON user_access(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_access_expires_at ON user_access(expires_at);
    """
    return sql


def create_transactions_table_sqlite():
    """
    SQL to create transactions table for SQLite
    """
    sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        paddle_transaction_id TEXT UNIQUE NOT NULL,
        user_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        price_id TEXT NOT NULL,
        amount REAL NOT NULL,
        currency TEXT DEFAULT 'USD',
        status TEXT DEFAULT 'pending',
        customer_email TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_paddle_transaction_id ON transactions(paddle_transaction_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
    CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);
    """
    return sql


def create_user_access_table_sqlite():
    """
    SQL to create user_access table for SQLite
    """
    sql = """
    CREATE TABLE IF NOT EXISTS user_access (
        user_id TEXT PRIMARY KEY,
        paid_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        analysis_ids TEXT DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_user_access_user_id ON user_access(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_access_expires_at ON user_access(expires_at);
    """
    return sql


# Migration version tracking
MIGRATION_VERSION = "1.0.0"
MIGRATION_NAME = "paddle_billing_integration"


def apply_migration_postgresql(db_connection):
    """
    Apply migration for PostgreSQL
    """
    print(f"Applying migration: {MIGRATION_NAME} v{MIGRATION_VERSION}")

    cursor = db_connection.cursor()

    try:
        # Create transactions table
        cursor.execute(create_transactions_table_sql())
        print("✓ Created/verified transactions table")

        # Create user_access table
        cursor.execute(create_user_access_table_sql())
        print("✓ Created/verified user_access table")

        db_connection.commit()
        print(f"✓ Migration {MIGRATION_NAME} v{MIGRATION_VERSION} applied successfully")

    except Exception as e:
        db_connection.rollback()
        print(f"✗ Migration failed: {str(e)}")
        raise


def apply_migration_sqlite(db_connection):
    """
    Apply migration for SQLite
    """
    print(f"Applying migration: {MIGRATION_NAME} v{MIGRATION_VERSION}")

    cursor = db_connection.cursor()

    try:
        # Create transactions table
        cursor.execute(create_transactions_table_sqlite())
        print("✓ Created/verified transactions table")

        # Create user_access table
        cursor.execute(create_user_access_table_sqlite())
        print("✓ Created/verified user_access table")

        db_connection.commit()
        print(f"✓ Migration {MIGRATION_NAME} v{MIGRATION_VERSION} applied successfully")

    except Exception as e:
        db_connection.rollback()
        print(f"✗ Migration failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    # This is a template file
    # In production, integrate with your actual database connection
    print("Database Migration Template for Paddle Billing Integration")
    print("\nTo use this migration:")
    print("1. Install database dependencies:")
    print("   - PostgreSQL: pip install psycopg2-binary")
    print("   - SQLite: pip install sqlite3 (included in stdlib)")
    print("\n2. Run with database connection:")
    print("   python migrations.py postgresql")
    print("   python migrations.py sqlite")
    print("\n3. For production, use Alembic for proper migration management")
