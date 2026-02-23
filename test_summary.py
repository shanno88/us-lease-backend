# test_summary.py
import asyncio
import sys

sys.path.insert(0, ".")
from routes.lease_routes import extract_lease_summary_llm

SAMPLE_LEASE = """
RESIDENTIAL LEASE AGREEMENT
This Lease Agreement is made on July 1, 2012 between ABC Properties LLC (Landlord) and John Smith (Tenant).
Property Address: 123 Main Street, Apt 4B, New York, NY 10001
Lease Term: This lease shall begin on July 1, 2012 and end on June 30, 2013.
Rent: The monthly rent shall be $685.00, payable on the 1st of each month.
Security Deposit: Tenant shall pay a security deposit of $685.00.
Late Fee: A late fee of $25.00 will be charged if rent is not received by the 5th day of the month.
Early Termination: Tenant may terminate this lease early by giving 30 days written notice and paying a penalty equal to one month's rent.
"""

async def test():
    result = await extract_lease_summary_llm(SAMPLE_LEASE)
    print("Result:", result)

asyncio.run(test())
