import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from typing import Optional

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 15 * 1024 * 1024


def validate_file(file: UploadFile) -> None:
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")


async def save_upload_file(file: UploadFile) -> Path:
    validate_file(file)

    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return file_path


def cleanup_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Failed to cleanup file {file_path}: {e}")


def cleanup_temp_files(older_than_seconds: int = 3600) -> None:
    try:
        import time

        current_time = time.time()
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > older_than_seconds:
                    file_path.unlink()
    except Exception as e:
        print(f"Warning: Failed to cleanup temp files: {e}")
