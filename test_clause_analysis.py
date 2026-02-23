import asyncio
import sys

sys.path.insert(0, ".")
from routes.lease_routes import get_bilingual_analysis_batch


async def test():
    clauses = [
        {"clause_text": "1", "risk_level": "safe"},
        {
            "clause_text": "$25 late fee plus $5 per day after the 4th",
            "risk_level": "caution",
        },
        {
            "clause_text": "Security deposit of $685 is non-refundable",
            "risk_level": "danger",
        },
    ]
    results = await get_bilingual_analysis_batch(clauses)
    for i, r in enumerate(results):
        print(f"--- Clause {i + 1}: {clauses[i]['clause_text'][:30]} ---")
        if r.get("_skip"):
            print("(SKIPPED - noise)")
        else:
            print(f"analysis_zh: {r.get('analysis_zh', 'N/A')}")
            print(f"suggestion_zh: {r.get('suggestion_zh', 'N/A')}")
        print()


asyncio.run(test())
