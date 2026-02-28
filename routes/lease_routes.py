from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import logging
import uuid
import json
import random
import re
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
from services.ocr_service import get_ocr_service, analyze_lease_via_deepseek
from services.pdf_service import get_pdf_service
from store import (
    USER_ACCESS_STORE,
    USER_FREE_ANALYSIS_STORE,
    ANALYSIS_STORE,
    QUICK_ANALYZE_RATE_LIMITS,
    IP_RATE_LIMITS,
    QUICK_CLAUSE_HISTORY,
    QUICK_ANALYZE_USER_LIMIT,
    QUICK_ANALYZE_IP_LIMIT,
    QUICK_ANALYZE_WINDOW,
    FULL_ANALYSIS_LEASE_LIMIT,
    QUICK_CLAUSE_HISTORY_LIMIT,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lease"])

_deepseek_client = None


def has_valid_paid_access(user_id: str) -> bool:
    if user_id not in USER_ACCESS_STORE:
        return False
    access = USER_ACCESS_STORE[user_id]
    if "expires_at" not in access:
        return False
    expires_at = datetime.fromisoformat(access["expires_at"])
    return datetime.now() < expires_at


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

Return ONLY valid JSON, no markdown, no explanations."""


KEY_CLAUSES_EXTRACTION_PROMPT = """You are a lease clause analyzer. Extract and analyze key clauses from the lease text.

YOUR TASK
Identify 5-10 of the most important clauses in this lease. For each clause, provide ALL fields in a single object.

OUTPUT FORMAT (STRICT JSON ARRAY)
Return a JSON array where each object has EXACTLY these fields:

[
  {
    "clause_id": "Clause 1",
    "clause_text_en": "The exact English text from the lease (max 300 chars)",
    "summary_zh": "Specific Chinese explanation of what this clause means and its impact on the tenant (2-3 sentences)",
    "risk_level": "low"
  },
  ...
]

RISK LEVEL RULES
- "low": Standard, tenant-friendly, or no significant concerns
- "medium": Some concerns, additional costs, or restrictions
- "high": Significant risk, unusual terms, or heavily favors landlord

SUMMARY_ZH RULES
- MUST be specific to THIS clause, not generic filler
- MUST describe what the clause regulates and its impact on tenant
- Include specific amounts, dates, or conditions when present
- NEVER use generic phrases like "该条款已分析潜在问题" or "这是标准条款"

CLAUSE_TEXT_EN RULES
- MUST be actual English text from the lease
- NEVER use "1", "2", "3" or any placeholder
- If text is longer than 300 chars, truncate but keep it meaningful

IMPORTANT CLAUSES TO LOOK FOR
1. Lease term (dates, duration)
2. Rent amount and payment terms
3. Security deposit
4. Late fees
5. Early termination / break lease
6. Utilities responsibility
7. Maintenance and repairs
8. Pet policy
9. Guest policy
10. Landlord entry rights
11. Subletting rules
12. Rent increase terms

Return ONLY the JSON array, no markdown, no explanations."""


def remove_duplicate_sentences(text: str) -> str:
    """Remove consecutive duplicate sentences from Chinese text."""
    if not text or not text.strip():
        return text

    sentences = re.split(r"([。！？\n])", text)
    cleaned = []
    prev_sentence = ""

    for i, part in enumerate(sentences):
        if part in "。！？\n":
            if cleaned and prev_sentence:
                cleaned.append(part)
            prev_sentence = ""
        else:
            part_stripped = part.strip()
            if part_stripped and part_stripped != prev_sentence:
                cleaned.append(part)
                prev_sentence = part_stripped

    return "".join(cleaned).strip()


async def extract_lease_summary_llm(full_text: str) -> Dict[str, Any]:
    """
    Extract structured lease summary using LLM.
    Returns a dict with monthly_rent_amount, currency, dates, etc.
    """
    if not full_text or not full_text.strip():
        return {}

    try:
        logger.info(f"[LLM SUMMARY] Starting extraction for {len(full_text)} chars")

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
        logger.info(f"[LLM SUMMARY] Raw LLM response: {raw_content[:500]}")

        clean = raw_content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        parsed = json.loads(clean)
        logger.info(
            f"[LLM SUMMARY] Parsed JSON: {json.dumps(parsed, indent=2, ensure_ascii=False)}"
        )
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"[LLM SUMMARY] JSON parse error: {e}")
        return {}
    except Exception as e:
        logger.error(f"[LLM SUMMARY] Extraction failed: {e}")
        return {}


async def get_chinese_explanation(english_text: str) -> Optional[str]:
    """Get Chinese explanation for English text."""
    if not english_text or not english_text.strip():
        return None

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
            return parsed[0].get("chinese_explanation", "")
        return None
    except Exception as e:
        logger.error(f"DeepSeek error in get_chinese_explanation: {e}")
        return None


def parse_bilingual_response(response_text: str) -> List[Dict[str, str]]:
    """Parse bilingual response from AI."""
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


async def extract_key_clauses_from_lease(full_text: str) -> List[Dict[str, Any]]:
    """
    Extract key clauses from lease text using a SINGLE AI call.
    Returns clauses with: clause_id, clause_text_en, summary_zh, risk_level
    """
    if not full_text or not full_text.strip():
        logger.warning("[KEY_CLAUSES] Empty full_text provided")
        return []

    try:
        logger.info(f"[KEY_CLAUSES] Extracting clauses from {len(full_text)} chars")

        response = get_deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": KEY_CLAUSES_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract key clauses from this lease:\n\n{full_text[:6000]}",
                },
            ],
            temperature=0.1,
            max_tokens=4000,
        )

        raw_content = response.choices[0].message.content or "[]"
        logger.info(f"[KEY_CLAUSES] Raw AI response: {raw_content[:500]}")

        clean = raw_content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        parsed = json.loads(clean)

        if not isinstance(parsed, list):
            logger.error(f"[KEY_CLAUSES] Expected list, got {type(parsed)}")
            return []

        clauses = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            clause_text_en = str(item.get("clause_text_en", "")).strip()

            # Skip if clause_text_en is invalid (just numbers, too short, or placeholder)
            if not clause_text_en or len(clause_text_en) < 20:
                continue
            if clause_text_en.isdigit() or clause_text_en in ["1", "2", "3", "4", "5"]:
                continue
            if clause_text_en.lower() in ["clause", "section", "page"]:
                continue

            risk_level = str(item.get("risk_level", "medium")).lower()
            if risk_level not in ["low", "medium", "high"]:
                risk_level = "medium"

            clause = {
                "clause_id": str(item.get("clause_id") or f"Clause {idx + 1}"),
                "clause_text_en": clause_text_en[:500],  # Limit length
                "summary_zh": str(item.get("summary_zh", ""))[:500],
                "risk_level": risk_level,
            }
            clauses.append(clause)

        logger.info(f"[KEY_CLAUSES] Extracted {len(clauses)} valid clauses")
        return clauses

    except json.JSONDecodeError as e:
        logger.error(f"[KEY_CLAUSES] JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"[KEY_CLAUSES] Extraction failed: {e}")
        return []


async def generate_sample_clauses(
    full_text: str, fast_mode: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Generate clause analyses from the lease text.
    Simple, direct mapping: ONE AI call -> direct field mapping.

    Returns:
        Tuple of (clauses list, ai_duration in seconds)
    """
    ai_start_time = time.time()

    clauses = await extract_key_clauses_from_lease(full_text)

    # Limit number of clauses based on mode
    max_clauses = 3 if fast_mode else 10
    clauses = clauses[:max_clauses]

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

    if not has_valid_paid_access(user_id):
        logger.warning(f"Access denied for user_id: {user_id}")
        raise HTTPException(
            status_code=403,
            detail="Access denied. Please purchase the lease analysis service to use this feature.",
        )

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

        # Use the full extraction pipeline from ocr_service
        lease_data = analyze_lease_via_deepseek(full_text)

        print("===== ANALYZE ENDPOINT: lease_data received =====", flush=True)
        print(f"lease_data keys: {list(lease_data.keys())}", flush=True)
        print(f"clauses count: {len(lease_data.get('clauses', []))}", flush=True)
        if lease_data.get("clauses"):
            print(f"first clause: {lease_data['clauses'][0]}", flush=True)
        print("=================================================", flush=True)

        # Build key_info from lease_data
        key_info = {
            "rent": lease_data.get("rent"),
            "deposit": lease_data.get("deposit"),
            "term_months": lease_data.get("term_months"),
            "start_date": lease_data.get("start_date"),
            "end_date": lease_data.get("end_date"),
            "landlord": lease_data.get("landlord"),
            "tenant": lease_data.get("tenant"),
        }

        # Get clauses and summary from lease_data
        all_clauses = lease_data.get("clauses", [])
        summary_text = lease_data.get("summary", "")

        # Extract high-risk clauses
        high_risk_clauses = [c for c in all_clauses if c.get("risk_level") == "high"]

        has_full_access = True  # TEST MODE: always unlocked

        processing_time = time.time() - start_time

        logger.info(f"Lease analysis completed successfully in {processing_time:.2f}s")

        analysis_id = str(uuid.uuid4())
        ANALYSIS_STORE[analysis_id] = {
            "full_text": full_text,
            "key_info": key_info,
            "summary": summary_text,
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
            "llmDuration": 0,
            "businessDuration": round(processing_time * 1000),
            "totalDuration": round(processing_time, 2),
        }

        return {
            "success": True,
            "data": {
                "analysis_id": analysis_id,
                "full_text": full_text,
                "key_info": key_info,
                "summary": summary_text,
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
