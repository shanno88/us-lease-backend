"""
Global in-memory stores for the application.
This module centralizes shared state to avoid circular imports.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta


# User access store: tracks 30-day pass access
# Structure: { user_id: { "expires_at": "ISO date string", "analysis_ids": [...] } }
USER_ACCESS_STORE: Dict[str, Dict[str, Any]] = {}

# Free analysis tracking: tracks if user has used their free analysis
# Structure: { user_id: True/False }
USER_FREE_ANALYSIS_STORE: Dict[str, bool] = {}

# Analysis results store: stores full analysis results by ID
# Structure: { analysis_id: { ... full analysis data ... } }
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}

# Rate limiting for quick-analyze endpoint
QUICK_ANALYZE_RATE_LIMITS: Dict[str, List[datetime]] = {}
IP_RATE_LIMITS: Dict[str, List[datetime]] = {}

# Quick clause history storage (user_id -> list of results)
QUICK_CLAUSE_HISTORY: Dict[str, List[Dict[str, Any]]] = {}

# Time windows
QUICK_ANALYZE_WINDOW = timedelta(hours=24)

# Limits
QUICK_ANALYZE_USER_LIMIT = 3
QUICK_ANALYZE_IP_LIMIT = 20
FULL_ANALYSIS_LEASE_LIMIT = 5  # Max 5 leases per 30-day pass
QUICK_CLAUSE_HISTORY_LIMIT = 3
