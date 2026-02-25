"""
FastAPI application: PDF upload -> Markdown conversion with GPU (Docling).
"""
import logging
import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.converter import convert_pdf_to_markdown, convert_report

_log = logging.getLogger(__name__)
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "markdown"
CACHE_TTL_SECONDS = 60 * 60


def _setup_file_logging() -> None:
    """Append app logs to logs/app.log while keeping console logs."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    target_file = str(LOG_FILE.resolve())

    # Avoid duplicate file handlers on reload/import.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == target_file:
            return

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root_logger.addHandler(file_handler)
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)


def _cleanup_expired_cache_files() -> None:
    """Remove cached markdown files older than the configured TTL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for cached_file in CACHE_DIR.glob("*.md"):
        try:
            if now - cached_file.stat().st_mtime > CACHE_TTL_SECONDS:
                cached_file.unlink(missing_ok=True)
                (CACHE_DIR / f"{cached_file.stem}.json").unlink(missing_ok=True)
        except OSError as e:
            _log.warning("Failed to cleanup cache file %s: %s", cached_file, e)


app = FastAPI(
    title="Docling PDF to Markdown",
    description="Upload PDF, get Markdown. Uses Docling with GPU acceleration.",
    version="0.1.0",
)

# Optional: set debug logging for conversion details
_setup_file_logging()
logging.getLogger("app").setLevel(logging.DEBUG)
_log.info("File logging enabled: %s", LOG_FILE)


def _ensure_pdf(filename: str) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="File must be a PDF (.pdf)")


@app.get("/health")
def health():
    """Health check: CUDA available and pipeline ready."""
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass
    # Pipeline is ready after first conversion or can be lazy-init on first request
    return {"cuda_available": cuda_available, "status": "ok"}


@app.post("/convert")
async def convert(
    request: Request,
    file: UploadFile = File(..., description="PDF file to convert"),
):
    """
    Convert uploaded PDF to Markdown and return downloadable file metadata.

    Returns a JSON list where each item contains:
    - id: generated file id
    - url: download URL
    - name: markdown filename
    """
    _ensure_pdf(file.filename or "")
    _cleanup_expired_cache_files()
    request_start = time.perf_counter()

    try:
        content = await file.read()
    except Exception as e:
        _log.warning("Failed to read upload: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        markdown, summary = convert_pdf_to_markdown(tmp_path, return_summary=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    request_duration = time.perf_counter() - request_start
    _log.info("Request total duration_sec=%.3f", request_duration)
    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info("Conversion summary: %s", summary_dict)

    filename = (file.filename or "document.pdf").rsplit(".", 1)[0] + ".md"
    file_id = uuid4().hex
    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"
    cache_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"name": filename}, ensure_ascii=False),
        encoding="utf-8",
    )
    download_url = str(request.url_for("download_file", file_id=file_id))

    return JSONResponse(
        content=[
            {
                "id": file_id,
                "url": download_url,
                "name": filename,
            }
        ],
    )


@app.post("/convert_report")
async def convert_report_api(
    request: Request,
    file: UploadFile = File(..., description="Financial report PDF to convert"),
):
    """
    Convert uploaded financial-report PDF using report-specific preprocessing.
    """
    _ensure_pdf(file.filename or "")
    _cleanup_expired_cache_files()
    request_start = time.perf_counter()

    try:
        content = await file.read()
    except Exception as e:
        _log.warning("Failed to read upload: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        markdown, summary = convert_report(tmp_path, return_summary=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    request_duration = time.perf_counter() - request_start
    _log.info("Report request total duration_sec=%.3f", request_duration)
    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info("Report conversion summary: %s", summary_dict)

    filename = (file.filename or "document.pdf").rsplit(".", 1)[0] + ".md"
    file_id = uuid4().hex
    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"
    cache_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"name": filename}, ensure_ascii=False),
        encoding="utf-8",
    )
    download_url = str(request.url_for("download_file", file_id=file_id))

    return JSONResponse(
        content=[
            {
                "id": file_id,
                "url": download_url,
                "name": filename,
            }
        ],
    )


@app.get("/files/{file_id}", name="download_file")
def download_file(file_id: str):
    """Download converted markdown by file id."""
    _cleanup_expired_cache_files()
    if not file_id or not file_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid file id")

    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"
    if not cache_path.is_file():
        raise HTTPException(status_code=404, detail="File not found or expired")

    download_name = cache_path.name
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict) and isinstance(metadata.get("name"), str):
                download_name = metadata["name"]
        except Exception as e:
            _log.warning("Failed to read metadata for %s: %s", file_id, e)

    return FileResponse(
        path=cache_path,
        media_type="text/markdown",
        filename=download_name,
    )
