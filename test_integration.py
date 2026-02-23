"""
Integration test for Step 3: Full lease analysis with summary and high_risk_clauses.
"""

import asyncio
import sys
import json

sys.path.insert(0, ".")

from utils.text_parser import (
    validate_summary_response,
    build_key_info_from_summary,
    filter_and_extract_high_risk_clauses,
)
from routes.lease_routes import extract_lease_summary_llm, get_bilingual_analysis_batch

SAMPLE_LEASE = """
RESIDENTIAL LEASE AGREEMENT

This Lease Agreement is made on July 1, 2012 between ABC Properties LLC (Landlord) and John Smith (Tenant).

Property Address: 123 Main Street, Apt 4B, New York, NY 10001

Lease Term: This lease shall begin on July 1, 2012 and end on June 30, 2013.

Rent: The monthly rent shall be $685.00, payable on the 1st of each month.

Security Deposit: Tenant shall pay a security deposit of $685.00 which is non-refundable.

Late Fee: A late fee of $25.00 will be charged if rent is not received by the 5th day of the month, plus $5 per day thereafter.

Early Termination: Tenant may terminate this lease early by giving 30 days written notice and paying a penalty equal to two months rent.

Utilities: Tenant is responsible for all utilities including electricity, gas, and internet.

Pet Policy: No pets allowed without landlord's written consent. A pet deposit of $200 is required if approved.

Maintenance: Tenant shall keep the premises in clean and sanitary condition.

Entry: Landlord may enter the premises with 24 hours notice for inspection or repairs.

"""

SAMPLE_CLAUSES = [
    {"clause_number": 1, "clause_text": "1", "risk_level": "safe"},
    {"clause_number": 2, "clause_text": "2", "risk_level": "safe"},
    {
        "clause_number": 3,
        "clause_text": "Security deposit of $685 is non-refundable",
        "risk_level": "danger",
    },
    {
        "clause_number": 4,
        "clause_text": "$25 late fee plus $5 per day after the 4th",
        "risk_level": "caution",
    },
    {
        "clause_number": 5,
        "clause_text": "Early termination requires 30 days notice and 2 months rent penalty",
        "risk_level": "danger",
    },
    {
        "clause_number": 6,
        "clause_text": "Tenant is responsible for all utilities",
        "risk_level": "caution",
    },
    {
        "clause_number": 7,
        "clause_text": "No pets without landlord consent",
        "risk_level": "safe",
    },
]


async def test_full_integration():
    print("=" * 60)
    print("STEP 3 INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Extract summary from LLM
    print("\n1. EXTRACTING SUMMARY FROM LLM...")
    raw_summary = await extract_lease_summary_llm(SAMPLE_LEASE)
    summary = validate_summary_response(raw_summary)

    print("\n   RAW LLM RESPONSE:")
    print(json.dumps(raw_summary, indent=2, ensure_ascii=False))

    print("\n   VALIDATED SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Step 2: Build key_info from summary (with fallback logic)
    print("\n2. BUILDING KEY_INFO FROM SUMMARY...")
    if summary.get("monthly_rent_amount") or summary.get("lease_start_date"):
        key_info = build_key_info_from_summary(summary)
        print("   Using LLM-based key_info")
    else:
        print("   Would fallback to regex-based extraction")
        key_info = build_key_info_from_summary(summary)

    print("\n   KEY_INFO:")
    print(json.dumps(key_info, indent=2, ensure_ascii=False))

    # Step 3: Filter clauses and extract high-risk
    print("\n3. FILTERING CLAUSES AND EXTRACTING HIGH-RISK...")
    filtered_clauses, high_risk_clauses = filter_and_extract_high_risk_clauses(
        SAMPLE_CLAUSES
    )

    print(f"\n   ORIGINAL CLAUSES: {len(SAMPLE_CLAUSES)}")
    print(f"   FILTERED CLAUSES: {len(filtered_clauses)}")
    print(f"   HIGH RISK CLAUSES: {len(high_risk_clauses)}")

    print("\n   FILTERED CLAUSES:")
    for c in filtered_clauses:
        print(f"      [{c['risk_level']:8}] {c['clause_text'][:50]}")

    print("\n   HIGH RISK CLAUSES:")
    for c in high_risk_clauses:
        print(f"      [{c['risk_level']:8}] {c['clause_text'][:50]}")

    # Step 4: Final response structure
    print("\n4. FINAL API RESPONSE STRUCTURE:")
    response = {
        "success": True,
        "data": {
            "key_info": key_info,
            "summary": summary,
            "clauses": filtered_clauses,
            "high_risk_clauses": high_risk_clauses,
            "total_clauses": len(filtered_clauses),
        },
    }

    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_integration())
