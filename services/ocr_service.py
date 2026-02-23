import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional


from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR


from config import settings


logger = logging.getLogger(__name__)


DEFAULT_LEASE_DATA = {
    "rent": None,
    "deposit": None,
    "term_months": None,
    "start_date": None,
    "end_date": None,
    "landlord": None,
    "tenant": None,
    "risk_score": 50,
    "risk_level": "medium",
    "red_flags": ["Could not parse lease details"],
    "negotiation_tips": ["Review document manually"],
    "summary": "Analysis incomplete due to parsing error.",
    "clauses": [],
}

REGEX_PATTERNS = {
    "rent": [
        r"(?:rent|monthly\s*rent|base\s*rent|sum\s*of)[:\s]*[\$￥]?\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s*month|/mo|monthly)?",
        r"[\$￥]\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s*month|/mo|monthly)",
        r"([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars?)\s*(?:per\s*month|monthly)",
        r"(?:pay|paying)\s*[\$￥]?\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s*month|monthly)?",
    ],
    "deposit": [
        r"(?:security\s*deposit|deposit)[:\s]*[\$￥]?\s*([\d,]+(?:\.\d{2})?)",
        r"押金[:\s]*[\$￥]?\s*([\d,]+(?:\.\d{2})?)",
    ],
    "term_months": [
        r"(?:term|lease|period|duration)\s*(?::|of|is)?\s*(\d{1,2})\s*(?:month|months)",
        r"(\d{1,2})\s*(?:month|months)\s*(?:term|lease|period)",
        r"(?:for|term\s*of)\s*(\d{1,2})\s*(?:year|yr)(?:\s*(?:term|lease))?",
        r"beginning[^.]+ending[^.]+(\d{4})",  # Extract year range from "beginning X ending Y"
    ],
    "start_date": [
        r"(?:beginning|start|commencing|effective)[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        r"(?:beginning|start|commencing|effective)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:beginning|start|commencing|effective)[:\s]+(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
    ],
    "landlord": [
        r"(?:landlord|lessor|owner)[:\s]+([A-Z][A-Za-z\s]+?)(?=\s*(?:,|\.|\n|Tenant|$))",
    ],
    "tenant": [
        r"(?:tenant|lessee|renter)[:\s]+([A-Z][A-Za-z\s]+?)(?=\s*(?:,|\.|\n|$))",
    ],
}

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def normalize_amount(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[\$￥,\s]", "", str(value))
    return cleaned


def normalize_date(value: str) -> str:
    if not value:
        return ""
    value = str(value).strip()

    match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", value)
    if match:
        month_name, day, year = match.groups()
        month = MONTH_MAP.get(month_name.lower(), 0)
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"

    match = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", value)
    if match:
        m, d, y = match.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{int(m):02d}-{int(d):02d}"

    match = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
    if match:
        y, m, d = match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"

    return value


def normalize_term_months(value: str, pattern: str = "") -> int:
    if not value:
        return 0
    try:
        num = int(re.sub(r"[^\d]", "", str(value)))
        if "year" in pattern.lower():
            return num * 12
        return num
    except (ValueError, TypeError):
        return 0


def regex_extract_fallback(full_text: str) -> dict:
    extracted = {}

    for field, patterns in REGEX_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if field == "term_months":
                    if "year" in pattern.lower():
                        try:
                            extracted[field] = int(value) * 12
                        except ValueError:
                            continue
                    else:
                        try:
                            extracted[field] = int(value)
                        except ValueError:
                            continue
                elif field in ("rent", "deposit"):
                    extracted[field] = f"${value}"
                else:
                    extracted[field] = value
                logger.info(f"[REGEX_FALLBACK] Extracted {field}: {extracted[field]}")
                break

    return extracted


def extract_json_from_llm_response(raw_content: str) -> Optional[dict]:
    if not raw_content:
        logger.warning("[JSON_EXTRACT] Empty raw_content received")
        return None

    clean = raw_content.strip()
    logger.info(
        f"[JSON_EXTRACT] Attempting to extract JSON from response (length={len(clean)})"
    )
    logger.debug(f"[JSON_EXTRACT] Raw content preview: {clean[:300]}...")

    try:
        result = json.loads(clean)
        logger.info("[JSON_EXTRACT] Direct parse succeeded")
        return result
    except json.JSONDecodeError:
        pass

    if clean.startswith("```json"):
        clean = clean[7:].lstrip("\n")
    elif clean.startswith("```"):
        first_newline = clean.find("\n")
        if first_newline != -1:
            clean = clean[first_newline + 1 :]
        else:
            clean = clean[3:]

    if clean.endswith("```"):
        last_backticks = clean.rfind("```")
        clean = clean[:last_backticks].rstrip()

    clean = clean.strip()

    try:
        result = json.loads(clean)
        logger.info("[JSON_EXTRACT] Parse succeeded after markdown stripping")
        return result
    except json.JSONDecodeError:
        pass

    brace_start = clean.find("{")
    brace_end = clean.rfind("}")

    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_candidate = clean[brace_start : brace_end + 1]
        try:
            result = json.loads(json_candidate)
            logger.info(
                "[JSON_EXTRACT] Parse succeeded by extracting JSON object from response"
            )
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"[JSON_EXTRACT] Found braces but parse failed: {e}")

    json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    matches = re.findall(json_pattern, clean, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            logger.info("[JSON_EXTRACT] Parse succeeded via regex extraction")
            return result
        except json.JSONDecodeError:
            continue

    logger.error(
        f"[JSON_EXTRACT] All extraction methods failed. Response preview: {clean[:500]}"
    )
    return None


LEASE_EXTRACTION_PROMPT = """You are a precise lease data extraction system. Extract fields from lease text and output ONLY valid JSON.

FIELD DEFINITIONS:
- rent: Monthly rent as a plain number string, e.g., "685" (remove $ and commas)
- deposit: Security deposit amount as a plain number string, e.g., "685" 
- term_months: Lease duration in months as integer, e.g., 12. Convert "1 year" to 12, "2 years" to 24.
- start_date: Lease start date as YYYY-MM-DD string, e.g., "2012-07-01"
- end_date: Lease end date as YYYY-MM-DD string, e.g., "2013-06-30"
- landlord: Landlord/owner name as string, e.g., "Robert Johnson"
- tenant: Tenant/renter name as string, e.g., "Mary Williams"
- risk_score: Integer 0-100 (0-40=low risk, 41-70=medium, 71-100=high)
- risk_level: String "low", "medium", or "high"
- red_flags: Array of strings describing potential issues
- negotiation_tips: Array of strings with advice
- summary: 2-3 sentence summary string
- clauses: Array of clause objects (see CLAUSE EXTRACTION below)

EXTRACTION RULES:
1. NEVER return null for rent/deposit/term_months/start_date/landlord/tenant if ANY related text exists
2. Make reasonable estimates when exact values are unclear
3. Convert currency formats: "$1,500.00" → "1500", "US$685" → "685"
4. Convert date formats: "July 1, 2012" → "2012-07-01", "7/1/2012" → "2012-07-01"
5. Calculate term_months from date range if explicit term not found
6. Look for names after labels: "Landlord:", "Tenant:", "Lessor:", "Lessee:"
7. Output ONLY raw JSON, no markdown code blocks, no explanations

CLAUSE EXTRACTION:
Identify ALL major clauses/sections in the lease. For each clause, create an object with:
- id: "clause_1", "clause_2", etc.
- number: The clause number if present, e.g., "1.", "2.1", "3." (or empty string if none)
- category: One of: "lease_term", "rent_and_payment", "deposit", "maintenance", "early_termination", "rent_increase", "use_and_pets", "landlord_entry", "liability_and_insurance", "dispute_resolution", "other"
- title_en: The heading text in English, e.g., "LEASE TERM", "RENT", "SECURITY DEPOSIT"
- original_text: The key English sentence(s) from that clause (max 200 chars)
- summary_zh: A concise Chinese explanation of what this clause means (1-2 sentences)
- risk_level: "low", "medium", or "high" based on tenant-friendliness

IMPORTANT FOR CLAUSES:
- Include EVERY important clause, even if risk is low (do not skip low-risk clauses)
- Extract clauses by identifying headings, numbered sections, or distinct paragraphs
- summary_zh must be in Chinese (中文), explaining the clause's effect in plain language

FEW-SHOT EXAMPLE:
Input lease text:
---
LEASE AGREEMENT
Landlord: Robert Johnson  
Tenant: Mary Williams
1. LEASE TERM: Tenants agree to lease this dwelling for a fixed term of one year, beginning July 1, 2012 and ending June 30, 2013.
2. RENT: The rent is the sum of $685 per month, payable on the first day of each month.
3. SECURITY DEPOSIT: Tenant shall pay a security deposit of $685 upon signing.
---
Output JSON:
{"rent": "685", "deposit": "685", "term_months": 12, "start_date": "2012-07-01", "end_date": "2013-06-30", "landlord": "Robert Johnson", "tenant": "Mary Williams", "risk_score": 25, "risk_level": "low", "red_flags": [], "negotiation_tips": ["Consider requesting longer notice period for termination"], "summary": "Standard 12-month residential lease with $685 monthly rent and matching security deposit. Low risk agreement.", "clauses": [{"id": "clause_1", "number": "1.", "category": "lease_term", "title_en": "LEASE TERM", "original_text": "Tenants agree to lease this dwelling for a fixed term of one year, beginning July 1, 2012 and ending June 30, 2013.", "summary_zh": "这一条说明租期为 12 个月，从 2012-07-01 到 2013-06-30。", "risk_level": "low"}, {"id": "clause_2", "number": "2.", "category": "rent_and_payment", "title_en": "RENT", "original_text": "The rent is the sum of $685 per month, payable on the first day of each month.", "summary_zh": "这一条说明每月租金为 685 美元，需在每月第一天支付。", "risk_level": "low"}, {"id": "clause_3", "number": "3.", "category": "deposit", "title_en": "SECURITY DEPOSIT", "original_text": "Tenant shall pay a security deposit of $685 upon signing.", "summary_zh": "这一条说明签约时需支付 685 美元的押金。", "risk_level": "low"}]}"""


def analyze_lease_via_deepseek(full_text: str) -> dict:
    text_len = len(full_text) if full_text else 0
    text_preview = full_text[:1000] if full_text else ""
    text_preview = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text_preview)
    text_preview = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text_preview)
    logger.info(f"[DIAGNOSTIC] === FULL_TEXT ANALYSIS ===")
    logger.info(f"[DIAGNOSTIC] full_text length: {text_len} chars")
    logger.info(f"[DIAGNOSTIC] full_text preview (redacted): {text_preview[:500]}")
    if not full_text or not full_text.strip():
        logger.warning("[DEEPSEEK] Empty full_text, returning DEFAULT_LEASE_DATA")
        return DEFAULT_LEASE_DATA.copy()
    if not settings.DEEPSEEK_API_KEY:
        logger.error("[DEEPSEEK] DEEPSEEK_API_KEY is not configured")
        return DEFAULT_LEASE_DATA.copy()
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
    )
    try:
        logger.info(f"[DEEPSEEK] Calling LLM with {len(full_text)} chars of text")
        user_message = f"""Now extract from this lease text:
{full_text[:7500]}
Remember: Output ONLY raw JSON, no markdown, no explanations. Fill ALL fields if any related text exists."""
        response = client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": LEASE_EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        raw_content = response.choices[0].message.content or ""
        logger.info(f"[DIAGNOSTIC] === LLM RESPONSE ===")
        logger.info(f"[DIAGNOSTIC] LLM response length: {len(raw_content)} chars")
        logger.info(f"[DIAGNOSTIC] LLM response raw: {raw_content}")
        parsed = extract_json_from_llm_response(raw_content)
        logger.info(f"[DIAGNOSTIC] === DEEPSEEK PARSED KEY_INFO (before merge) ===")
        if parsed:
            logger.info(
                f"[DIAGNOSTIC] rent={parsed.get('rent')}, deposit={parsed.get('deposit')}, "
                f"term_months={parsed.get('term_months')}, start_date={parsed.get('start_date')}, "
                f"landlord={parsed.get('landlord')}, tenant={parsed.get('tenant')}"
            )
        else:
            logger.info("[DIAGNOSTIC] Parsed JSON is None")
        if parsed is None:
            logger.error(
                "[DEEPSEEK] JSON extraction returned None, falling back to DEFAULT_LEASE_DATA"
            )
            return DEFAULT_LEASE_DATA.copy()
        result = DEFAULT_LEASE_DATA.copy()
        for key in result.keys():
            if key in parsed and parsed[key] is not None and parsed[key] != "":
                result[key] = parsed[key]
                logger.debug(f"[DEEPSEEK] Field '{key}': {parsed[key]}")
            else:
                logger.warning(
                    f"[DEEPSEEK] Field '{key}' missing/null/empty in LLM response"
                )
        regex_fallbacks = regex_extract_fallback(full_text)
        logger.info(f"[DIAGNOSTIC] === REGEX FALLBACK RESULTS ===")
        logger.info(f"[DIAGNOSTIC] Regex extracted: {regex_fallbacks}")
        for field in ["rent", "deposit", "term_months", "start_date"]:
            if result.get(field) is None and field in regex_fallbacks:
                result[field] = regex_fallbacks[field]
                logger.info(
                    f"[DIAGNOSTIC] Applied regex fallback for {field}: {regex_fallbacks[field]}"
                )
        if result.get("rent") is not None:
            rent_val = str(result["rent"])
            result["rent"] = re.sub(r"[^\d.]", "", rent_val)
        if result.get("deposit") is not None:
            deposit_val = str(result["deposit"])
            result["deposit"] = re.sub(r"[^\d.]", "", deposit_val)
        if result.get("risk_score") is None:
            result["risk_score"] = 50
        if result.get("risk_level") not in ("low", "medium", "high"):
            if isinstance(result.get("risk_score"), (int, float)):
                score = result["risk_score"]
                if score <= 40:
                    result["risk_level"] = "low"
                elif score <= 70:
                    result["risk_level"] = "medium"
                else:
                    result["risk_level"] = "high"
            else:
                result["risk_level"] = "medium"
        logger.info(f"[DIAGNOSTIC] === FINAL RESULT (after merge with defaults) ===")
        logger.info(
            f"[DIAGNOSTIC] key_info: rent={result.get('rent')}, deposit={result.get('deposit')}, "
            f"term_months={result.get('term_months')}, start_date={result.get('start_date')}, "
            f"landlord={result.get('landlord')}, tenant={result.get('tenant')}"
        )
        logger.info(
            f"[DIAGNOSTIC] risk_score={result.get('risk_score')}, risk_level={result.get('risk_level')}"
        )
        if "clauses" not in result or result["clauses"] is None:
            result["clauses"] = []
        clauses = result.get("clauses", [])
        logger.info(f"[DIAGNOSTIC] CLAUSES COUNT: {len(clauses)}")
        if clauses:
            first_clause = clauses[0]
            logger.info(
                f"[DIAGNOSTIC] FIRST CLAUSE: category={first_clause.get('category')}, "
                f"title_en={first_clause.get('title_en')}, summary_zh={first_clause.get('summary_zh')}"
            )
        return result
    except json.JSONDecodeError as e:
        logger.error(f"[DEEPSEEK] JSON decode error: {e}")
        return DEFAULT_LEASE_DATA.copy()
    except Exception as e:
        logger.error(f"[DEEPSEEK] Lease analysis failed: {type(e).__name__}: {e}")
        return DEFAULT_LEASE_DATA.copy()


class OCRService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.ocr = RapidOCR()

    def recognize_image(self, image_path: Path) -> List[Dict[str, Any]]:
        try:
            result, _ = self.ocr(str(image_path))

            lines = []
            for item in result or []:
                box, text, confidence = item
                lines.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "box": [[int(p[0]), int(p[1])] for p in box],
                    }
                )
            return lines
        except Exception as e:
            logger.error(f"RapidOCR failed for {image_path}: {e}")
            return []

    def recognize_images(self, image_paths: List[Path]) -> Dict[str, Any]:
        all_lines = []
        page_texts = []

        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i + 1}/{len(image_paths)}: {image_path}")

            lines = self.recognize_image(Path(image_path))
            if lines:
                all_lines.extend(lines)
                page_text = "\n".join(line["text"] for line in lines)
                page_texts.append(page_text)

        full_text = "\n\n".join(page_texts)

        return {
            "lines": all_lines,
            "full_text": full_text,
            "page_count": len(image_paths),
        }

    async def extract_lease_from_file(
        self, image_paths: List[str], user_id: str = "unknown"
    ) -> Dict[str, Any]:
        print("=== SHANNO TEST LOG: extract_lease_from_file CALLED ===", flush=True)
        print(f"[DIAGNOSTIC] image_paths count: {len(image_paths)}", flush=True)
        logger.info("=== SHANNO TEST LOG: extract_lease_from_file CALLED ===")
        logger.info(f"[DIAGNOSTIC] === extract_lease_from_file START ===")
        logger.info(
            f"[DIAGNOSTIC] user_id: {user_id}, image_paths count: {len(image_paths)}"
        )

        ocr_result = self.recognize_images([Path(p) for p in image_paths])
        full_text = ocr_result.get("full_text", "")

        print(f"[DIAGNOSTIC] === OCR RESULT ===", flush=True)
        print(f"[DIAGNOSTIC] full_text length: {len(full_text)} chars", flush=True)
        print(f"[DIAGNOSTIC] full_text preview:\n{full_text[:800]}", flush=True)
        logger.info(f"[DIAGNOSTIC] === OCR RESULT ===")
        logger.info(f"[DIAGNOSTIC] full_text length: {len(full_text)} chars")
        logger.info(f"[DIAGNOSTIC] full_text preview: {full_text[:500]}")

        if not full_text or not full_text.strip():
            logger.error("[DIAGNOSTIC] No text extracted from document")
            raise Exception("No text extracted from document")

        lease_data = analyze_lease_via_deepseek(full_text)

        print(f"[DIAGNOSTIC] === FINAL KEY_INFO ===", flush=True)
        print(
            f"[DIAGNOSTIC] rent={lease_data.get('rent')}, deposit={lease_data.get('deposit')}",
            flush=True,
        )
        print(
            f"[DIAGNOSTIC] term_months={lease_data.get('term_months')}, start_date={lease_data.get('start_date')}",
            flush=True,
        )
        print(
            f"[DIAGNOSTIC] landlord={lease_data.get('landlord')}, tenant={lease_data.get('tenant')}",
            flush=True,
        )
        print(
            f"[DIAGNOSTIC] CLAUSES COUNT: {len(lease_data.get('clauses', []))}",
            flush=True,
        )
        logger.info(f"[DIAGNOSTIC] === FINAL RESPONSE DATA ===")
        logger.info(f"[DIAGNOSTIC] lease_data keys: {list(lease_data.keys())}")
        logger.info(
            f"[DIAGNOSTIC] key_info: rent={lease_data.get('rent')}, deposit={lease_data.get('deposit')}, "
            f"term_months={lease_data.get('term_months')}, start_date={lease_data.get('start_date')}, "
            f"landlord={lease_data.get('landlord')}, tenant={lease_data.get('tenant')}"
        )
        logger.info(f"[DIAGNOSTIC] CLAUSES COUNT: {len(lease_data.get('clauses', []))}")

        return {
            "success": True,
            "data": {
                "analysis_id": str(uuid.uuid4()),
                "has_full_access": False,
                "risk_score": lease_data["risk_score"],
                "risk_level": lease_data["risk_level"],
                "red_flags": lease_data["red_flags"],
                "negotiation_tips": lease_data["negotiation_tips"],
                "summary": lease_data["summary"],
                "key_info": {
                    "rent": lease_data["rent"],
                    "deposit": lease_data["deposit"],
                    "term_months": lease_data["term_months"],
                    "start_date": lease_data["start_date"],
                    "end_date": lease_data["end_date"],
                    "landlord": lease_data["landlord"],
                    "tenant": lease_data["tenant"],
                },
                "clauses": lease_data.get("clauses", []),
                "full_text": full_text,
                "page_count": ocr_result["page_count"],
            },
        }


_ocr_service = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
