"""
FastAPI application: PDF upload -> Markdown conversion with GPU (Docling).
Optimized with async processing and content-based caching.
"""
import asyncio
import hashlib
import logging
import json
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from app.converter import convert_pdf_to_markdown, convert_report

_log = logging.getLogger(__name__)
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "markdown"
CACHE_TTL_SECONDS = 60 * 60

# 单线程池：保证同一时间只有一个 GPU 转换任务运行，
# 其余请求在 FastAPI async 层并发接收文件 / 命中缓存直接返回。
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-convert")

# 内存缓存：content_hash -> (file_id, timestamp)
_content_cache: Dict[str, tuple[str, float]] = {}

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _setup_file_logging() -> None:
    """Append app logs to logs/app.log and ensure console (stderr) has a clear format."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    target_file = str(LOG_FILE.resolve())

    # Avoid duplicate file handlers on reload/import.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == target_file:
            break
    else:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FMT))
        root_logger.addHandler(file_handler)

    # 统一 console 格式，便于 docker logs 查看
    import sys
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr:
            h.setFormatter(logging.Formatter(_FMT))
            break
    else:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(_FMT))
        root_logger.addHandler(console)
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)


def _cleanup_expired_cache_files() -> None:
    """Remove cached markdown files older than the configured TTL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    
    # 清理文件缓存
    for cached_file in CACHE_DIR.glob("*.md"):
        try:
            if now - cached_file.stat().st_mtime > CACHE_TTL_SECONDS:
                cached_file.unlink(missing_ok=True)
                (CACHE_DIR / f"{cached_file.stem}.json").unlink(missing_ok=True)
        except OSError as e:
            _log.warning("Failed to cleanup cache file %s: %s", cached_file, e)
    
    # 清理内存缓存
    expired_hashes = [
        h for h, (_, ts) in _content_cache.items()
        if now - ts > CACHE_TTL_SECONDS
    ]
    for h in expired_hashes:
        del _content_cache[h]


def _compute_content_hash(content: bytes) -> str:
    """计算文件内容的SHA256哈希值"""
    return hashlib.sha256(content).hexdigest()


def _check_cache(content_hash: str) -> Optional[str]:
    """
    检查缓存是否存在且有效
    返回file_id或None
    """
    if content_hash not in _content_cache:
        return None
    
    file_id, timestamp = _content_cache[content_hash]
    cache_path = CACHE_DIR / f"{file_id}.md"
    
    # 验证文件仍然存在且未过期
    if cache_path.is_file() and (time.time() - timestamp) < CACHE_TTL_SECONDS:
        return file_id
    
    # 缓存失效，清除
    del _content_cache[content_hash]
    return None


def _save_to_cache(content_hash: str, file_id: str, markdown: str, filename: str) -> None:
    """保存转换结果到缓存"""
    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"
    
    cache_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"name": filename, "content_hash": content_hash}, ensure_ascii=False),
        encoding="utf-8",
    )
    
    # 更新内存缓存
    _content_cache[content_hash] = (file_id, time.time())


app = FastAPI(
    title="Docling PDF to Markdown",
    description="Upload PDF, get Markdown. Uses Docling with GPU acceleration.",
    version="0.2.0",
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
    return {"cuda_available": cuda_available, "status": "ok"}


@app.post("/convert")
async def convert(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to convert"),
):
    """
    Convert uploaded PDF to Markdown and return downloadable file metadata.
    Uses content-based caching to avoid re-converting identical files.

    Returns a JSON list where each item contains:
    - id: generated file id
    - url: download URL
    - name: markdown filename
    - cached: whether result was from cache
    """
    _ensure_pdf(file.filename or "")
    background_tasks.add_task(_cleanup_expired_cache_files)
    request_start = time.perf_counter()

    try:
        content = await file.read()
    except Exception as e:
        _log.warning("convert failed | read error: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if not content:
        _log.warning("convert rejected | empty file")
        raise HTTPException(status_code=400, detail="Empty file")

    size_kb = len(content) / 1024
    filename = (file.filename or "document.pdf").rsplit(".", 1)[0] + ".md"
    
    # 计算内容哈希
    content_hash = _compute_content_hash(content)
    
    # 检查缓存
    cached_file_id = _check_cache(content_hash)
    if cached_file_id:
        request_duration = time.perf_counter() - request_start
        download_url = str(request.url_for("download_file", file_id=cached_file_id))
        _log.info(
            "convert cached | filename=%s size=%.1fKB hash=%s duration=%.3fs",
            filename, size_kb, content_hash[:16], request_duration,
        )
        return JSONResponse(
            content=[
                {
                    "id": cached_file_id,
                    "url": download_url,
                    "name": filename,
                    "cached": True,
                }
            ],
        )

    # 异步转换（在线程池中执行，避免阻塞事件循环）
    tmp_path = Path(tempfile.mktemp(suffix=".pdf"))
    tmp_path.write_bytes(content)

    try:
        # 在线程池中执行转换（使用partial包装关键字参数）
        loop = asyncio.get_event_loop()
        convert_func = partial(convert_pdf_to_markdown, pdf_path=tmp_path, return_summary=True)
        markdown, summary = await loop.run_in_executor(_executor, convert_func)
    except FileNotFoundError as e:
        _log.warning("convert failed | filename=%s: %s", file.filename or "(unnamed)", e)
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        _log.warning("convert failed | filename=%s: %s", file.filename or "(unnamed)", e)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    request_duration = time.perf_counter() - request_start
    file_id = uuid4().hex
    
    # 保存到缓存
    _save_to_cache(content_hash, file_id, markdown, filename)
    
    download_url = str(request.url_for("download_file", file_id=file_id))

    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info(
        "convert done | filename=%s size=%.1fKB hash=%s duration=%.3fs",
        filename, size_kb, content_hash[:16], request_duration,
    )

    return JSONResponse(
        content=[
            {
                "id": file_id,
                "url": download_url,
                "name": filename,
                "cached": False,
            }
        ],
    )


@app.post("/convert_report")
async def convert_report_api(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Financial report PDF to convert"),
):
    """
    Convert uploaded financial-report PDF using report-specific preprocessing.
    Uses content-based caching to avoid re-converting identical files.
    """
    _ensure_pdf(file.filename or "")
    background_tasks.add_task(_cleanup_expired_cache_files)
    request_start = time.perf_counter()

    try:
        content = await file.read()
    except Exception as e:
        _log.warning("convert_report failed | read error: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if not content:
        _log.warning("convert_report rejected | empty file")
        raise HTTPException(status_code=400, detail="Empty file")

    size_kb = len(content) / 1024
    filename = (file.filename or "document.pdf").rsplit(".", 1)[0] + ".md"
    
    # 计算内容哈希（加上"report:"前缀以区分普通转换）
    content_hash = "report:" + _compute_content_hash(content)
    
    # 检查缓存
    cached_file_id = _check_cache(content_hash)
    if cached_file_id:
        request_duration = time.perf_counter() - request_start
        download_url = str(request.url_for("download_file", file_id=cached_file_id))
        _log.info(
            "convert_report cached | filename=%s size=%.1fKB hash=%s duration=%.3fs",
            filename, size_kb, content_hash[:23], request_duration,
        )
        return JSONResponse(
            content=[
                {
                    "id": cached_file_id,
                    "url": download_url,
                    "name": filename,
                    "cached": True,
                }
            ],
        )

    # 异步转换
    tmp_path = Path(tempfile.mktemp(suffix=".pdf"))
    tmp_path.write_bytes(content)

    try:
        loop = asyncio.get_event_loop()
        convert_func = partial(convert_report, pdf_path=tmp_path, return_summary=True)
        markdown, summary = await loop.run_in_executor(_executor, convert_func)
    except FileNotFoundError as e:
        _log.warning("convert_report failed | filename=%s: %s", file.filename or "(unnamed)", e)
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        _log.warning("convert_report failed | filename=%s: %s", file.filename or "(unnamed)", e)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    request_duration = time.perf_counter() - request_start
    file_id = uuid4().hex
    
    # 保存到缓存
    _save_to_cache(content_hash, file_id, markdown, filename)
    
    download_url = str(request.url_for("download_file", file_id=file_id))

    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info(
        "convert_report done | filename=%s size=%.1fKB hash=%s duration=%.3fs",
        filename, size_kb, content_hash[:23], request_duration,
    )

    return JSONResponse(
        content=[
            {
                "id": file_id,
                "url": download_url,
                "name": filename,
                "cached": False,
            }
        ],
    )


@app.get("/files/{file_id}", name="download_file")
def download_file(file_id: str):
    """Download converted markdown by file id."""
    if not file_id or not file_id.isalnum():
        _log.warning("download rejected | invalid file_id=%s", file_id)
        raise HTTPException(status_code=400, detail="Invalid file id")

    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"
    if not cache_path.is_file():
        _log.warning("download 404 | file_id=%s", file_id)
        raise HTTPException(status_code=404, detail="File not found or expired")

    download_name = cache_path.name
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict) and isinstance(metadata.get("name"), str):
                download_name = metadata["name"]
        except Exception as e:
            _log.warning("download | failed to read metadata for %s: %s", file_id, e)

    return FileResponse(
        path=cache_path,
        media_type="text/markdown",
        filename=download_name,
    )


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理线程池"""
    _executor.shutdown(wait=True)
