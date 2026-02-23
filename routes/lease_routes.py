from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import logging
import uuid
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from openai import OpenAI

from config import settings
from utils.file_handler import save_upload_file, cleanup_file
from utils.text_parser import (
    extract_key_info,
    validate_summary_response,
    build_key_info_from_summary,
    filter_and_extract_high_risk_clauses,
)
from services.ocr_service import get_ocr_service
from services.pdf_service import get_pdf_service

logger = logging.getLogger(__name__)

_deepseek_client = None


def get_deepseek_client() -> OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        if not settings.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
        _deepseek_client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1"
        )
    return _deepseek_client


BILINGUAL_EXPLAINER_SYSTEM_PROMPT = """You are a rental agreement explainer for Chinese international students in the US.

YOUR TASK
Convert any English lease-related text into a bilingual two-line format.

IMPORTANT
The English text you receive may be:
- an original lease clause, OR
- an English analysis, suggestion, or recommendation (e.g. "Negotiate pet fee waiver or one-time $200 instead of monthly", "Save ~$100/year").

In ALL cases, you must treat each English line as content to be bilingualized.

OUTPUT FORMAT (STRICT – NO DEVIATIONS)

For each English line you receive in the user message, output exactly two lines:

- Line 1: Copy the English text EXACTLY as provided in the input.
- Line 2: Start with "中文解释：" and then write 1–3 sentences of natural Chinese explaining:
  - what that English line means,
  - what the tenant should do or understand,
  - and, if relevant, the money impact or risk.

Put ONE blank line between different English lines.

RULES
1. Line 1 = always copy the English input line exactly. Never modify it.
2. Line 2 = always start with "中文解释：" and be written mainly in Chinese.
3. Do not add extra titles, emojis, or bullet points.
4. For multiple lines in one message, output repeated blocks:

[original English line]
中文解释：[Chinese explanation]

separated by one blank line.

RESPONSE BEHAVIOR
- As soon as you receive English text, immediately output the two-line blocks.
- Do not reply with "Understood" or "Ready"."""

BILINGUAL_ANALYSIS_SYSTEM_PROMPT = """You are a rental agreement analyst for Chinese international students in the US.

YOUR TASK
For each lease clause provided, generate bilingual analysis and suggestion.

INPUT FORMAT
You will receive clauses in this format:
---CLAUSE---
[clause text]
---RISK---
[risk level: safe/caution/danger]
---END---

NOISE FILTERING (IMPORTANT)
If a clause is obviously just noise (section numbers like "1", "2", "Section 5", page markers, pure formatting), output:
{"skip": true}
The backend will filter these out. Focus your analysis on substantive clauses only.

OUTPUT FORMAT (STRICT JSON)
For substantive clauses, output a JSON object with these exact fields:
{
  "analysis_en": "1-2 sentences: what this clause means and concerns",
  "analysis_zh": "1-2 sentences in Chinese following the template below",
  "suggestion_en": "1 sentence with practical advice",
  "suggestion_zh": "1 sentence in Chinese following the template below"
}

CHINESE OUTPUT TEMPLATES

analysis_zh template (2 parts):
1. First, briefly describe what the clause regulates
2. Then, explicitly state the impact on the tenant (financial risk, flexibility loss, rights limitation)

Example structure: "该条款规定了[内容]，对租客的影响是[具体后果/风险]。"
- Focus on WORST-CASE scenarios when relevant
- Mention specific dollar amounts or time limits if present

suggestion_zh template:
Give 1-2 practical, actionable steps using phrases like:
- "务必提前确认..." (Make sure to confirm in advance...)
- "如果不接受，可以跟房东协商..." (If unacceptable, negotiate with landlord...)
- "建议记录在书面合同中..." (Recommend documenting in written contract...)
- "注意保留..." (Keep records of...)
- "签约前建议..." (Before signing, consider...)

RULES
1. Chinese output should be practical and actionable, not generic filler
2. Focus on tenant's financial risk, rights, and flexibility
3. Be specific: mention amounts, deadlines, conditions when present
4. For safe clauses, still explain what it does and confirm it's standard
5. Output ONLY valid JSON, no markdown
6. For multiple clauses, output a JSON array: [{...}, {...}]"""

LEASE_SUMMARY_SYSTEM_PROMPT = """You are a lease document analyst. Extract structured information from lease text.

YOUR TASK
Extract the following fields from the lease document. Return ONLY valid JSON with these exact fields:

{
  "monthly_rent_amount": <number or null>,
  "currency": "<USD or other currency code>",
  "lease_start_date": "<ISO date YYYY-MM-DD or null>",
  "lease_end_date": "<ISO date YYYY-MM-DD or null>",
  "lease_duration_months": <integer or null>,
  "security_deposit_amount": <number or null>,
  "landlord_name": "<name or null>",
  "tenant_name": "<name or null>",
  "late_fee_summary_zh": "<one Chinese sentence describing late fee rule, or '未明确写明滞纳金条款'>",
  "early_termination_risk_zh": "<one Chinese sentence about early termination risk, or '未明确写明提前解约条款'>",
  "overall_risk": "<low|medium|high>"
}

FIELD RULES
- monthly_rent_amount: Extract the numeric monthly rent (e.g., 685). If only weekly/annual given, convert to monthly. Look for patterns like "Rent: $685/month", "Monthly Rent: 685", "Base Rent: $685".
- currency: Usually "USD" for US leases
- lease_start_date: Format as YYYY-MM-DD. Convert "July 1, 2012" to "2012-07-01". Look for "Commencement Date", "Start Date", "Effective Date", "begins on".
- lease_end_date: Format as YYYY-MM-DD. Look for "Expiration Date", "End Date", "terminates on", "ends on".
- lease_duration_months: Calculate from dates if not explicitly stated. 12 months = 1 year. Look for "term of 12 months", "one year lease", "12-month term".
- security_deposit_amount: Usually equals 1 month rent, but extract actual value if stated. Look for "Security Deposit", "Deposit", "refundable deposit".
- landlord_name: Full name or company name of landlord/lessor. Look for "Landlord:", "Lessor:", "Owner:", after "between" clauses.
- tenant_name: Full name of tenant/lessee. Look for "Tenant:", "Lessee:", "Renter:", after "between" clauses.
- late_fee_summary_zh: Example: "滞纳金为每日5美元，从到期日后第5天开始计算"
- early_termination_risk_zh: Example: "提前解约需支付2个月租金作为违约金" or warn if clause is missing
- overall_risk: "low"=standard lease, "medium"=some concerning terms, "high"=significant risks

RULES
1. Return ONLY valid JSON, no markdown or explanations
2. Use null for fields not found in the document - DO NOT use "N/A", "Not found", or empty strings
3. Be precise with dates and numbers - extract actual values from the text
4. Chinese summaries should be natural and concise (1 sentence each)
5. Calculate overall_risk based on: deposit amount, termination terms, fee structures
6. CRITICAL: Always extract landlord_name and tenant_name from signature blocks or party identification sections
7. For rent and deposit amounts, extract ONLY the numeric value (no $ sign, no currency symbol)"""


def parse_bilingual_response(response_text: str) -> List[Dict[str, str]]:
    clauses = []
    blocks = [b.strip() for b in response_text.split("\n\n") if b.strip()]

    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        if len(lines) >= 2:
            english_line = lines[0].strip()
            chinese_line = next((l for l in lines if l.startswith("中文解释：")), None)

            if chinese_line:
                clauses.append(
                    {
                        "clause_text": english_line,
                        "chinese_explanation": chinese_line.replace(
                            "中文解释：", ""
                        ).strip(),
                    }
                )

    return clauses


async def get_chinese_explanation(english_text: str) -> Optional[str]:
    try:
        response = get_deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": BILINGUAL_EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": english_text},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        result = response.choices[0].message.content or ""
        parsed = parse_bilingual_response(result)

        if parsed:
            return parsed[0]["chinese_explanation"]
        return None
    except Exception as e:
        logger.error(f"DeepSeek error in get_chinese_explanation: {e}")
        return None


async def extract_lease_summary_llm(full_text: str) -> Dict[str, Any]:
    """
    Extract structured lease summary using LLM.
    Returns a dict with monthly_rent_amount, currency, dates, etc.
    This function does NOT modify any existing code paths.
    """
    raw_content = ""
    try:
        logger.info(f"[LLM SUMMARY] Starting extraction for {len(full_text)} chars")
        logger.info(f"[LLM SUMMARY] First 500 chars of input: {full_text[:500]}")

        response = get_deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": LEASE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": full_text[:8000]},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        raw_content = response.choices[0].message.content or "{}"
        logger.info(
            f"[LLM SUMMARY] Raw LLM response (first 500 chars): {raw_content[:500]}"
        )
        logger.info(f"[LLM SUMMARY] Raw LLM response (full): {raw_content}")

        clean = raw_content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        logger.info(f"[LLM SUMMARY] Cleaned JSON string: {clean}")

        parsed = json.loads(clean)
        logger.info(
            f"[LLM SUMMARY] Parsed JSON: {json.dumps(parsed, indent=2, ensure_ascii=False)}"
        )

        for key in [
            "monthly_rent_amount",
            "security_deposit_amount",
            "lease_duration_months",
            "landlord_name",
            "tenant_name",
            "lease_start_date",
            "lease_end_date",
        ]:
            value = parsed.get(key)
            if value is None or value == "" or value == "N/A" or value == "Not found":
                logger.warning(
                    f"[LLM SUMMARY] Field '{key}' has missing/invalid value: {value}"
                )
            else:
                logger.info(
                    f"[LLM SUMMARY] Field '{key}' extracted successfully: {value}"
                )

        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"[LLM SUMMARY] JSON parse error: {e}")
        logger.error(
            f"[LLM SUMMARY] Raw content that failed: {raw_content[:1000] if raw_content else 'empty'}"
        )
        return {}
    except Exception as e:
        logger.error(f"[LLM SUMMARY] Extraction failed: {e}")
        return {}


# In-memory storage for analysis results
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}
USER_ACCESS_STORE: Dict[str, Dict[str, Any]] = {}
USER_FREE_ANALYSIS_STORE: Dict[str, bool] = {}

# Rate limiting storage for quick-analyze endpoint
QUICK_ANALYZE_RATE_LIMITS: Dict[str, List[datetime]] = {}
IP_RATE_LIMITS: Dict[str, List[datetime]] = {}

# Quick clause history storage (user_id -> list of results)
QUICK_CLAUSE_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
QUICK_CLAUSE_HISTORY_LIMIT = 3

# Rate limiting constants
QUICK_ANALYZE_USER_LIMIT = 3
QUICK_ANALYZE_IP_LIMIT = 20
QUICK_ANALYZE_WINDOW = timedelta(hours=24)

# Full lease analysis pass constants
FULL_ANALYSIS_LEASE_LIMIT = 5  # Max 5 leases per 30-day pass
FULL_ANALYSIS_ACCESS_DURATION = timedelta(days=30)  # 30-day access window


async def get_chinese_explanation_batch(texts: List[str]) -> List[str]:
    """
    Get Chinese explanations for multiple texts in a single AI call.
    Returns a list of Chinese explanations in the same order as input texts.
    """
    if not texts:
        return []

    combined_input = "\n\n".join(texts)

    try:
        response = get_deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": BILINGUAL_EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": combined_input},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        result = response.choices[0].message.content or ""
        parsed = parse_bilingual_response(result)

        explanations = []
        for i, text in enumerate(texts):
            if i < len(parsed):
                explanations.append(parsed[i].get("chinese_explanation", ""))
            else:
                explanations.append("")

        return explanations
    except Exception as e:
        logger.error(f"DeepSeek error in get_chinese_explanation_batch: {e}")
        return [""] * len(texts)


async def get_bilingual_analysis_batch(
    clause_data: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Get bilingual analysis and suggestion for multiple clauses in a single AI call.
    Input: list of dicts with 'clause_text' and 'risk_level' keys
    Returns: list of dicts with analysis_en, analysis_zh, suggestion_en, suggestion_zh
    """
    if not clause_data:
        return []

    combined_input = ""
    for i, clause in enumerate(clause_data):
        combined_input += f"""---CLAUSE---
{clause.get("clause_text", "")}
---RISK---
{clause.get("risk_level", "safe")}
---END---

"""

    try:
        response = get_deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": BILINGUAL_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": combined_input},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        result = response.choices[0].message.content or ""

        # Try to parse JSON response
        try:
            # Remove potential markdown code blocks
            clean_result = result.strip()
            if clean_result.startswith("```"):
                clean_result = (
                    clean_result.split("\n", 1)[1]
                    if "\n" in clean_result
                    else clean_result
                )
            if clean_result.endswith("```"):
                clean_result = clean_result.rsplit("```", 1)[0]
            clean_result = clean_result.strip()

            parsed = json.loads(clean_result)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    if isinstance(item, dict):
                        if item.get("skip"):
                            results.append({"_skip": True})
                        else:
                            results.append(item)
                    else:
                        results.append({})
                return results
            elif isinstance(parsed, dict):
                if parsed.get("skip"):
                    return [{"_skip": True}]
                return [parsed]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AI response as JSON: {result[:200]}")

        # Fallback: return empty dicts
        return [{}] * len(clause_data)
    except Exception as e:
        logger.error(f"DeepSeek error in get_bilingual_analysis_batch: {e}")
        return [{}] * len(clause_data)


router = APIRouter(tags=["lease"])


def classify_clause(clause_text: str) -> str:
    """
    Classify a clause into 'meta', 'core_term', or 'other'.
    """
    text_lower = clause_text.lower()

    # Meta: titles, headers, page markers, party names, dates
    meta_patterns = [
        "--- page",
        "sample rental agreement",
        "residential lease agreement",
        "this agreement made this",
        "this lease agreement",
        "between the parties",
        "landlord:",
        "tenant:",
        "lessee:",
        "lessor:",
        "date:",
        "property address:",
        "page ",
        "section",
    ]

    for pattern in meta_patterns:
        if pattern in text_lower:
            # Additional check: if very short and looks like a header
            if len(clause_text) < 50 and (
                clause_text.strip().startswith("---")
                or "page" in text_lower
                or clause_text.strip().endswith(":")
            ):
                return "meta"
            # Check if it's primarily a title/header
            if (
                any(p in text_lower for p in ["agreement", "lease", "contract"])
                and len(clause_text) < 100
            ):
                return "meta"

    # Core terms: important economic/legal terms
    core_term_keywords = [
        "rent",
        "deposit",
        "security deposit",
        "late fee",
        "term",
        "lease term",
        "utilities",
        "maintenance",
        "repair",
        "termination",
        "eviction",
        "sublet",
        "sublease",
        "pet",
        "guest",
        "occupant",
        "parking",
        "insurance",
        "entry",
        "access",
        "notice",
        "renewal",
        "break lease",
        "early termination",
        "grace period",
        "payment",
        "monthly",
        "annual",
        "yearly",
        "prorate",
        "pro-rate",
    ]

    for keyword in core_term_keywords:
        if keyword in text_lower:
            return "core_term"

    return "other"


async def generate_sample_clauses(
    full_text: str, fast_mode: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    """Generate sample clause analyses from the lease text with bilingual analysis and Chinese explanations

    Args:
        full_text: The lease text to analyze
        fast_mode: If True, generates only 3 clauses with simplified analysis (for preview mode)

    Returns:
        Tuple of (clauses list, ai_duration in seconds)
    """
    ai_start_time = time.time()
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

    if len(paragraphs) < 5:
        sentences = [s.strip() for s in full_text.split(".") if s.strip()]
        paragraphs = sentences

    if fast_mode:
        num_clauses = min(len(paragraphs), 3)
    else:
        num_clauses = min(len(paragraphs), random.randint(15, 20))

    risk_levels = ["safe", "caution", "danger"]
    risk_weights = [0.5, 0.35, 0.15]

    clause_data_for_analysis = []
    clause_texts_for_explanation = []

    for i in range(num_clauses):
        paragraph_idx = i % len(paragraphs)
        clause_text = paragraphs[paragraph_idx][:200]
        risk_level = random.choices(risk_levels, weights=risk_weights)[0]
        category = classify_clause(clause_text)

        clause_data_for_analysis.append(
            {
                "clause_text": clause_text,
                "risk_level": risk_level,
                "category": category,
                "clause_number": i + 1,
            }
        )
        clause_texts_for_explanation.append(clause_text)

    # Get bilingual analysis/suggestion from AI
    bilingual_analyses = await get_bilingual_analysis_batch(clause_data_for_analysis)

    # Get Chinese explanations for clause text
    chinese_explanations = await get_chinese_explanation_batch(
        clause_texts_for_explanation
    )

    # Build final clause objects
    clauses = []
    for i, data in enumerate(clause_data_for_analysis):
        bilingual = bilingual_analyses[i] if i < len(bilingual_analyses) else {}

        if bilingual.get("_skip"):
            continue

        analysis_en = bilingual.get(
            "analysis_en", "This clause has been analyzed for potential concerns."
        )
        analysis_zh = bilingual.get("analysis_zh", "该条款已分析潜在问题。")
        suggestion_en = bilingual.get(
            "suggestion_en", "Review this clause carefully before signing."
        )
        suggestion_zh = bilingual.get("suggestion_zh", "签署前请仔细阅读此条款。")

        clause = {
            "clause_number": data["clause_number"],
            "clause_text": data["clause_text"],
            "category": data["category"],
            "risk_level": data["risk_level"],
            "chinese_explanation": chinese_explanations[i]
            if i < len(chinese_explanations)
            else "",
            "analysis_en": analysis_en,
            "analysis_zh": analysis_zh,
            "suggestion_en": suggestion_en,
            "suggestion_zh": suggestion_zh,
            "analysis": analysis_en,
            "suggestion": suggestion_en,
        }
        clauses.append(clause)

    ai_duration = time.time() - ai_start_time
    return clauses, ai_duration


def check_user_access(user_id: str) -> Dict[str, Any]:
    """
    Check if user has valid 30-day access
    Returns: (has_access, expires_at, days_remaining, analyses_count, remaining_analyses)
    """
    if user_id not in USER_ACCESS_STORE:
        return {"has_access": False}

    access = USER_ACCESS_STORE[user_id]
    now = datetime.now()
    expires_at = datetime.fromisoformat(access["expires_at"])

    # Check if within 30-day window
    if now >= expires_at:
        return {"has_access": False}

    # Check analysis count (5-lease cap)
    analyses_count = len(access.get("analysis_ids", []))
    if analyses_count >= FULL_ANALYSIS_LEASE_LIMIT:
        return {
            "has_access": False,
            "reason": "lease_limit_reached",
            "message": f"You've used all {FULL_ANALYSIS_LEASE_LIMIT} analyses included in your 30-day pass. Please purchase another pass for more reviews.",
        }

    days_remaining = (expires_at - now).days
    remaining_analyses = FULL_ANALYSIS_LEASE_LIMIT - analyses_count

    return {
        "has_access": True,
        "expires_at": access["expires_at"],
        "days_remaining": days_remaining,
        "analyses_count": analyses_count,
        "remaining_analyses": remaining_analyses,
    }


@router.post("/analyze")
async def analyze_lease(
    files: List[UploadFile] = File(...),
    user_id: str = Query(..., description="User ID from frontend session"),
):
    start_time = time.time()
    temp_image_paths = []
    MAX_PAGES = 40

    try:
        logger.info(
            f"Starting lease analysis for {len(files)} file(s), user_id: {user_id}"
        )

        pdf_service = get_pdf_service()
        ocr_service = get_ocr_service()
        all_image_paths = []

        sorted_files = sorted(files, key=lambda f: f.filename or "")

        for file in sorted_files:
            uploaded_file_path = await save_upload_file(file)
            uploaded_path = Path(uploaded_file_path)
            temp_image_paths.append(uploaded_file_path)

            if pdf_service.is_pdf(uploaded_path):
                logger.info(f"Processing PDF file: {file.filename}")
                image_paths = pdf_service.pdf_to_images(uploaded_path)
                all_image_paths.extend(image_paths)
                temp_image_paths.extend(image_paths)
            elif pdf_service.is_image(uploaded_path):
                logger.info(f"Processing image file: {file.filename}")
                all_image_paths.append(uploaded_path)
            else:
                logger.error(f"Unsupported file format: {uploaded_path.suffix}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {uploaded_path.suffix}. Please upload PDF or image files.",
                )

        if not all_image_paths:
            logger.error("No pages found in the document(s)")
            raise HTTPException(
                status_code=400, detail="No pages found in the document(s)"
            )

        total_pages = len(all_image_paths)
        if total_pages > MAX_PAGES:
            logger.error(f"Too many pages: {total_pages}")
            raise HTTPException(
                status_code=400,
                detail=f"Your document has {total_pages} pages, which exceeds our limit of {MAX_PAGES} pages. Please upload a shorter lease or contact support.",
            )

        logger.info(f"Found {total_pages} page(s) to process from {len(files)} file(s)")
        ocr_result = ocr_service.recognize_images(all_image_paths)

        if not ocr_result or not ocr_result.get("full_text"):
            logger.error("OCR returned empty or invalid result")
            return {
                "success": False,
                "error": "No text extracted from document. The document may be empty, contain only images, or have formatting issues.",
            }

        full_text = ocr_result.get("full_text", "")

        if not full_text or not full_text.strip():
            logger.error("OCR returned empty text")
            return {
                "success": False,
                "error": "No text extracted from document. The document may be empty or contain only images.",
            }

        logger.info(f"Extracted {len(full_text)} characters from document")

        raw_summary = await extract_lease_summary_llm(full_text)
        logger.info(
            f"[VALIDATE] Raw summary before validation: {json.dumps(raw_summary, indent=2, ensure_ascii=False)}"
        )

        summary = validate_summary_response(raw_summary)
        logger.info(
            f"[VALIDATE] Summary after validation: {json.dumps(summary, indent=2, ensure_ascii=False)}"
        )

        if summary.get("monthly_rent_amount") or summary.get("lease_start_date"):
            key_info = build_key_info_from_summary(summary)
            logger.info(
                f"[KEY_INFO] Using LLM-based key_info: {json.dumps(key_info, indent=2, ensure_ascii=False)}"
            )
        else:
            key_info = extract_key_info(full_text)
            logger.info(
                f"[KEY_INFO] Using regex-based key_info as fallback: {json.dumps(key_info, indent=2, ensure_ascii=False)}"
            )

        all_clauses, ai_duration = await generate_sample_clauses(
            full_text, fast_mode=False
        )

        # Filter noise and extract high-risk clauses
        filtered_clauses, high_risk_clauses = filter_and_extract_high_risk_clauses(
            all_clauses
        )
        all_clauses = filtered_clauses

        if settings.should_bypass_test_user(user_id):
            logger.info(f"Test user bypass enabled for: {user_id}")
            has_full_access = True
        else:
            access_info = check_user_access(user_id)
            has_full_access = access_info["has_access"]

            if not has_full_access:
                reason = access_info.get("reason", "expired")
                if reason == "lease_limit_reached":
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "ACCESS_DENIED",
                            "message": f"您已使用完30天通行证中的全部{FULL_ANALYSIS_LEASE_LIMIT}次分析次数。如需继续使用，请购买新的通行证。",
                        },
                    )
                else:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "ACCESS_DENIED",
                            "message": "您当前没有有效的分析权限，请登录或完成支付后再试。",
                        },
                    )

        processing_time = time.time() - start_time

        logger.info(f"Lease analysis completed successfully in {processing_time:.2f}s")

        analysis_id = str(uuid.uuid4())
        ANALYSIS_STORE[analysis_id] = {
            "full_text": full_text,
            "key_info": key_info,
            "summary": summary,
            "all_clauses": all_clauses,
            "high_risk_clauses": high_risk_clauses,
            "lines": [
                {"text": line["text"], "confidence": line["confidence"]}
                for line in ocr_result.get("lines", [])
            ],
            "processing_time": round(processing_time, 2),
            "page_count": ocr_result.get("page_count", 0),
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
        }

        if user_id not in USER_ACCESS_STORE:
            USER_ACCESS_STORE[user_id] = {"analysis_ids": []}
        if analysis_id not in USER_ACCESS_STORE[user_id]["analysis_ids"]:
            USER_ACCESS_STORE[user_id]["analysis_ids"].append(analysis_id)

        logger.info(f"Stored analysis with ID: {analysis_id} for user: {user_id}")
        logger.info(
            f"User {user_id} has used {len(USER_ACCESS_STORE[user_id]['analysis_ids'])} of {FULL_ANALYSIS_LEASE_LIMIT} analyses"
        )

        timing_info = {
            "backendDuration": round(processing_time * 1000),
            "llmDuration": round(ai_duration * 1000),
            "businessDuration": round(processing_time * 1000),
            "totalDuration": round(processing_time, 2),
        }

        return {
            "success": True,
            "data": {
                "analysis_id": analysis_id,
                "full_text": full_text,
                "key_info": key_info,
                "summary": summary,
                "clauses": all_clauses,
                "high_risk_clauses": high_risk_clauses,
                "total_clauses": len(all_clauses),
                "shown_clauses": len(all_clauses),
                "max_clauses": len(all_clauses),
                "has_full_access": has_full_access,
                "user_id": user_id,
                "lines": [
                    {"text": line["text"], "confidence": line["confidence"]}
                    for line in ocr_result.get("lines", [])
                ],
                "processing_time": round(processing_time, 2),
                "page_count": ocr_result.get("page_count", 0),
                "timing": timing_info,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during lease analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to analyze lease: {str(e)}",
        }

    finally:
        for temp_path in temp_image_paths:
            cleanup_file(temp_path)


@router.post("/ocr")
async def ocr_lease(
    files: List[UploadFile] = File(...),
    user_id: str = Query("default_user", description="User ID from frontend"),
):
    """
    OCR endpoint: upload multiple PDF/JPG/PNG files → OCR + LLM analysis → return JSON

    Returns:
        {
            "success": true,
            "data": {
                "analysis_id": "uuid",
                "has_full_access": false,
                "risk_score": 65,
                "risk_level": "medium",
                "risk_clauses": [
                    {
                        "severity": "medium",
                        "clause_title_zh": "滞纳金条款",
                        "clause_text_original": "If rent is not received by the 5th of the month...",
                        "explanation_zh": "这一条款规定如果你在每月 5 号之后交房租，就要额外付当月房租 5% 的滞纳金。",
                        "suggestion_zh": "可以和房东协商改为固定金额（例如 50 美元封顶），避免随着房租上涨而无限增加滞纳金。"
                    }
                ],
                "summary": {
                    "monthly_rent_amount": 2000,
                    "currency": "USD",
                    "security_deposit_amount": 2000,
                    "lease_start_date": "2024-01-01",
                    "lease_end_date": "2024-12-31",
                    "lease_duration_months": 12,
                    "landlord_name": "John Smith",
                    "tenant_name": "Jane Doe",
                    "overall_risk": "medium",
                    "late_fee_summary_zh": "...",
                    "early_termination_risk_zh": "..."
                },
                "key_info": {
                    "rent_amount": "$2000/month",
                    "lease_term": "12 months",
                    "start_date": "2024-01-01",
                    "landlord": "John Smith",
                    "tenant": "Jane Doe"
                },
                "full_text": "...",
                "page_count": 5
            }
        }
    """
    start_time = time.time()
    temp_paths = []
    print("=== SHANNO TEST LOG: /api/lease/ocr handler CALLED ===", flush=True)
    print(f"=== SHANNO TEST LOG: files={len(files)}, user_id={user_id} ===", flush=True)
    logger.info("=== SHANNO TEST LOG: /api/lease/ocr handler CALLED ===")
    logger.info(f"=== SHANNO TEST LOG: files={len(files)}, user_id={user_id} ===")

    try:
        logger.info(f"Processing OCR request for {len(files)} file(s), user: {user_id}")

        if settings.should_bypass_test_user(user_id):
            logger.info(f"Test user bypass enabled for: {user_id}")
        else:
            access_info = check_user_access(user_id)
            has_paid_access = access_info.get("has_access", False)

            if not has_paid_access:
                free_analysis_used = USER_FREE_ANALYSIS_STORE.get(user_id, False)
                if free_analysis_used:
                    logger.info(f"Free analysis already used for user: {user_id}")
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "error": "FREE_LIMIT_REACHED",
                            "message": "You have already used your free analysis. A paid plan is required for additional reports.",
                        },
                    )
                logger.info(f"Using free analysis for user: {user_id}")
                USER_FREE_ANALYSIS_STORE[user_id] = True

        pdf_service = get_pdf_service()
        ocr_service = get_ocr_service()
        all_image_paths = []

        sorted_files = sorted(files, key=lambda f: f.filename or "")

        for file in sorted_files:
            if not file.filename:
                continue

            ext = Path(file.filename).suffix.lower()
            if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
                logger.warning(f"Skipping unsupported format: {file.filename}")
                continue

            logger.info(f"Processing file: {file.filename}")
            uploaded_path = await save_upload_file(file)
            temp_paths.append(str(uploaded_path))

            if pdf_service.is_pdf(Path(uploaded_path)):
                logger.info(f"Converting PDF to images: {file.filename}")
                image_paths = pdf_service.pdf_to_images(Path(uploaded_path))
                all_image_paths.extend(image_paths)
                temp_paths.extend([str(p) for p in image_paths])
                logger.info(f"Converted PDF to {len(image_paths)} page(s)")
            else:
                all_image_paths.append(Path(uploaded_path))

        if not all_image_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid files found. Please upload PDF or image files.",
            )

        logger.info(f"Total pages to process: {len(all_image_paths)}")

        result = await ocr_service.extract_lease_from_file(
            [str(p) for p in all_image_paths], user_id
        )

        result["data"]["processing_time"] = round(time.time() - start_time, 2)

        logger.info(f"OCR completed in {result['data']['processing_time']}s")
        logger.info(
            f"[OCR RESULT] risk_clauses: {json.dumps(result['data'].get('risk_clauses', []), indent=2, ensure_ascii=False)}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"OCR endpoint error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        for p in temp_paths:
            cleanup_file(p)


@router.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "service": "lease-ocr-api",
            "message": "API is running correctly",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/access")
async def check_lease_access(
    user_id: str = Query("default_user", description="User ID from frontend"),
):
    try:
        is_email_based = user_id.startswith("user_") and "@" in user_id
        is_logged_in = is_email_based

        free_analysis_used = USER_FREE_ANALYSIS_STORE.get(user_id, False)
        has_free_analysis_available = not free_analysis_used

        access_info = check_user_access(user_id)
        has_paid_access = access_info.get("has_access", False)

        return {
            "success": True,
            "is_logged_in": is_logged_in,
            "has_free_analysis_available": has_free_analysis_available,
            "has_active_plan": has_paid_access,
        }
    except Exception as e:
        logger.exception(f"Error checking lease access: {str(e)}")
        return {
            "success": False,
            "is_logged_in": False,
            "has_free_analysis_available": False,
            "has_active_plan": False,
            "error": str(e),
        }


@router.get("/full-report")
async def get_full_report(
    analysis_id: str = Query(..., description="Analysis ID from OCR"),
    user_id: str = Query(..., description="User ID from frontend session"),
):
    try:
        logger.info(
            f"Requesting full report for analysis_id: {analysis_id}, user_id: {user_id}"
        )

        if analysis_id not in ANALYSIS_STORE:
            logger.warning(f"Analysis ID not found: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail="Analysis not found. Please analyze a lease first.",
            )

        analysis = ANALYSIS_STORE[analysis_id]

        # Check if the analysis belongs to this user
        if analysis.get("user_id") != user_id:
            # Or check if the user has valid access
            access_info = check_user_access(user_id)
            if not access_info["has_access"]:
                logger.warning(
                    f"User {user_id} does not have access to analysis {analysis_id}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Access denied. This analysis belongs to another user or your access has expired.",
                )

        logger.info(f"Returning full report for analysis: {analysis_id}")

        return {
            "success": True,
            "data": {
                "analysis_id": analysis_id,
                "full_text": analysis["full_text"],
                "key_info": analysis["key_info"],
                "summary": analysis.get("summary", {}),
                "clauses": analysis["all_clauses"],
                "high_risk_clauses": analysis.get("high_risk_clauses", []),
                "total_clauses": len(analysis["all_clauses"]),
                "shown_clauses": len(analysis["all_clauses"]),
                "has_full_access": check_user_access(user_id)["has_access"],
                "user_id": user_id,
                "lines": analysis["lines"],
                "processing_time": analysis["processing_time"],
                "page_count": analysis["page_count"],
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error retrieving full report: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve full report: {str(e)}"
        )


def get_user_identifier(request: Request) -> str:
    """Get user identifier from header or fallback to IP-based ID"""
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return user_id
    # Fallback to IP-based ID (combine IP with a hash for some stability)
    client_host = request.client.host if request.client else "unknown"
    return f"ip_{client_host}"


def check_rate_limit(user_id: str, ip_address: str) -> Tuple[bool, int]:
    """
    Check if user/IP is within rate limits
    Returns: (allowed, remaining_attempts)
    """
    now = datetime.now()
    window_start = now - QUICK_ANALYZE_WINDOW

    # Clean up old entries
    if user_id in QUICK_ANALYZE_RATE_LIMITS:
        QUICK_ANALYZE_RATE_LIMITS[user_id] = [
            ts for ts in QUICK_ANALYZE_RATE_LIMITS[user_id] if ts > window_start
        ]
    if ip_address in IP_RATE_LIMITS:
        IP_RATE_LIMITS[ip_address] = [
            ts for ts in IP_RATE_LIMITS[ip_address] if ts > window_start
        ]

    # Check user limit
    user_requests = QUICK_ANALYZE_RATE_LIMITS.get(user_id, [])
    if len(user_requests) >= QUICK_ANALYZE_USER_LIMIT:
        remaining = 0
        logger.info(
            f"Rate limit exceeded for user {user_id}: {len(user_requests)} requests"
        )
        return (False, remaining)

    # Check IP limit
    ip_requests = IP_RATE_LIMITS.get(ip_address, [])
    if len(ip_requests) >= QUICK_ANALYZE_IP_LIMIT:
        remaining = max(0, QUICK_ANALYZE_USER_LIMIT - len(user_requests))
        logger.info(
            f"IP rate limit exceeded for {ip_address}: {len(ip_requests)} requests"
        )
        return (False, remaining)

    # Record this request
    if user_id not in QUICK_ANALYZE_RATE_LIMITS:
        QUICK_ANALYZE_RATE_LIMITS[user_id] = []
    QUICK_ANALYZE_RATE_LIMITS[user_id].append(now)

    if ip_address not in IP_RATE_LIMITS:
        IP_RATE_LIMITS[ip_address] = []
    IP_RATE_LIMITS[ip_address].append(now)

    remaining = QUICK_ANALYZE_USER_LIMIT - len(QUICK_ANALYZE_RATE_LIMITS[user_id])
    return (True, remaining)


def get_short_explanation(clause_text: str, risk_level: str) -> str:
    """
    Generate a brief 1-2 sentence explanation (simplified output)
    """
    text_lower = clause_text.lower()

    # High risk
    if risk_level == "danger":
        if "all" in text_lower and (
            "repair" in text_lower or "maintenance" in text_lower
        ):
            return "This clause shifts most repair costs to the tenant, regardless of fault."
        elif "enter" in text_lower and "any time" in text_lower:
            return "This allows landlord to enter your unit at any time without notice."
        elif "waive" in text_lower:
            return "This asks you to give up important legal rights as a tenant."
        return "This clause heavily favors the landlord and may limit your rights significantly."

    # Medium risk
    elif risk_level == "caution":
        if "late fee" in text_lower:
            return "Late fees are included—check amounts against your state's legal limits."
        elif "non-refundable" in text_lower:
            return "This fee may not be refundable—clarify what it covers."
        return (
            "This clause could lead to additional costs. Review the details carefully."
        )

    # Low risk
    return "This clause appears standard and doesn't raise obvious concerns."


def analyze_single_clause(clause_text: str) -> "tuple[str, str, str]":
    """
    Based on keyword rules to analyze a single clause
    Returns: (risk_level, analysis, suggestion)
    """
    text_lower = clause_text.lower()

    # High risk keywords
    high_risk_keywords = [
        "tenant responsible for all",
        "regardless of fault",
        "waive any right",
        "landlord may enter at any time",
        "no refund",
        "tenant liable for",
        "cannot terminate",
        "automatic renewal",
    ]

    # Medium risk keywords
    medium_risk_keywords = [
        "late fee",
        "additional charges",
        "landlord discretion",
        "may be charged",
        "tenant must pay",
        "non-refundable",
    ]

    # Check for high risk
    for keyword in high_risk_keywords:
        if keyword in text_lower:
            if "all" in text_lower and (
                "repair" in text_lower or "maintenance" in text_lower
            ):
                return (
                    "danger",
                    "This clause shifts all maintenance responsibility to you, regardless of fault. This is unusual and potentially unfair.",
                    "Request to limit your responsibility to damages caused by tenant negligence only. Standard leases don't make tenants responsible for normal wear and tear or structural issues.",
                )
            elif "enter" in text_lower and "any time" in text_lower:
                return (
                    "danger",
                    "This allows landlord unrestricted access to your apartment. Most jurisdictions require 24-48 hours notice except for emergencies.",
                    "Request specific language: 'Landlord may enter with 24-48 hours written notice, except in emergencies.'",
                )
            elif "waive" in text_lower:
                return (
                    "danger",
                    "Waiving rights can leave you without legal protection. This type of clause may not be enforceable in many states.",
                    "Consult a local tenant rights organization before signing. You may not be able to legally waive certain rights.",
                )
            else:
                return (
                    "danger",
                    "This clause contains language that may heavily favor landlord and limit your rights as a tenant.",
                    "Have a lawyer review this specific clause before signing, or request it be removed or modified.",
                )

    # Check for medium risk
    for keyword in medium_risk_keywords:
        if keyword in text_lower:
            if "late fee" in text_lower:
                return (
                    "caution",
                    "Late fees are common, but amounts should be reasonable. Check your state's laws on maximum late fee amounts.",
                    "Ensure there's a grace period (typically 3-5 days) and that fee doesn't exceed state limits (often $50 or 5% of rent).",
                )
            elif "non-refundable" in text_lower:
                return (
                    "caution",
                    "Non-refundable fees or deposits may not be legal in your state. Security deposits are typically refundable if you leave property in good condition.",
                    "Clarify what this fee covers and check local laws. Consider negotiating to make it refundable.",
                )
            else:
                return (
                    "caution",
                    "This clause may result in additional costs or give landlord significant discretion. Review carefully.",
                    "Ask for specific dollar amounts instead of vague terms like 'additional charges' or 'as determined by landlord.'",
                )

    # Default: appears safe
    return (
        "safe",
        "This clause appears standard and doesn't contain obvious red flags. However, it's always good to read the full context.",
        "Continue reviewing the complete lease for a comprehensive understanding. Our full analysis can check all clauses together.",
    )


# NOTE: /clause/quick-analyze is currently used only for internal testing.
# The public UI no longer exposes this feature.
@router.post("/clause/quick-analyze")
async def quick_analyze_clause(request: Request):
    """
    Quick analysis of a single clause (simplified preview, rate limited)
    Returns only risk_level and a brief explanation
    """
    try:
        data = await request.json()
        clause_text = data.get("clause_text", "").strip()

        if not clause_text:
            raise HTTPException(
                status_code=400,
                detail="Please paste only one short clause (max 250 characters) for the free preview.",
            )

        if len(clause_text) > 250:
            raise HTTPException(
                status_code=400,
                detail="This quick check is only for short clauses. Please paste a shorter sentence (up to 250 characters).",
            )

        user_id = get_user_identifier(request)
        ip_address = request.client.host if request.client else "unknown"

        allowed, remaining = check_rate_limit(user_id, ip_address)
        if not allowed:
            logger.warning(f"Rate limit exceeded for user {user_id}, IP {ip_address}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "limit_reached",
                    "message": "You've used your 3 free clause previews for today. For a full lease review, please upgrade to a paid report.",
                },
            )

        risk_level, full_analysis, suggestion = analyze_single_clause(clause_text)

        short_explanation = get_short_explanation(clause_text, risk_level)

        chinese_explanation = await get_chinese_explanation(short_explanation)

        risk_map = {"danger": "High", "caution": "Medium", "safe": "Low"}
        mapped_risk = risk_map.get(risk_level, "Medium")

        result = {
            "clause_text": clause_text,
            "risk_level": mapped_risk,
            "explanation_en": short_explanation,
            "explanation_zh": chinese_explanation or "",
            "created_at": datetime.now().isoformat(),
        }

        if user_id not in QUICK_CLAUSE_HISTORY:
            QUICK_CLAUSE_HISTORY[user_id] = []
        QUICK_CLAUSE_HISTORY[user_id].insert(0, result)
        QUICK_CLAUSE_HISTORY[user_id] = QUICK_CLAUSE_HISTORY[user_id][
            :QUICK_CLAUSE_HISTORY_LIMIT
        ]

        logger.info(
            f"Quick analyze - User: {user_id}, IP: {ip_address}, "
            f"Risk: {risk_level}, Remaining: {remaining}"
        )

        return {
            "success": True,
            "remaining_quota_today": remaining,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in quick analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clause/quick-analyze/history")
async def get_quick_clause_history(request: Request):
    """
    Get the last 3 quick clause analysis results for the user
    """
    try:
        user_id = get_user_identifier(request)
        history = QUICK_CLAUSE_HISTORY.get(user_id, [])

        return {
            "success": True,
            "history": history,
        }
    except Exception as e:
        logger.exception(f"Error fetching quick clause history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
