"""
FastAPI application: PDF upload / URL -> Markdown conversion with GPU (Docling).
Optimized with async processing and content-based caching.
"""

import asyncio
import hashlib
import logging
import json
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, unquote, parse_qs
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    BackgroundTasks,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.converter import convert_pdf_to_markdown, convert_report

_log = logging.getLogger(__name__)
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "markdown"
PDF_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "pdf"
CACHE_TTL_SECONDS = 60 * 60 * 24 * 180  # 180 天（约半年）

# 单线程池：保证同一时间只有一个 GPU 转换任务运行，
# 其余请求在 FastAPI async 层并发接收文件 / 命中缓存直接返回。
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-convert")

# 内存缓存：content_hash -> (file_id, timestamp)
_content_cache: Dict[str, tuple[str, float]] = {}

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# PDF 下载超时（秒）
_PDF_DOWNLOAD_TIMEOUT = 120


class UrlRequest(BaseModel):
    """JSON body for URL-based conversion."""

    url: str


def _setup_file_logging() -> None:
    """Append app logs to logs/app.log and ensure console (stderr) has a clear format."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    target_file = str(LOG_FILE.resolve())

    # Avoid duplicate file handlers on reload/import.
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", "") == target_file
        ):
            break
    else:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FMT))
        root_logger.addHandler(file_handler)

    # 统一 console 格式，便于 docker logs 查看
    import sys

    for h in root_logger.handlers:
        if (
            isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) is sys.stderr
        ):
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
    """Remove cached markdown and PDF files older than the configured TTL."""
    now = time.time()

    # 清理 markdown 缓存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for cached_file in CACHE_DIR.glob("*.md"):
        try:
            if now - cached_file.stat().st_mtime > CACHE_TTL_SECONDS:
                cached_file.unlink(missing_ok=True)
                (CACHE_DIR / f"{cached_file.stem}.json").unlink(missing_ok=True)
        except OSError as e:
            _log.warning("Failed to cleanup cache file %s: %s", cached_file, e)

    # 清理 PDF 缓存
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for cached_pdf in PDF_CACHE_DIR.glob("*.pdf"):
        try:
            if now - cached_pdf.stat().st_mtime > CACHE_TTL_SECONDS:
                cached_pdf.unlink(missing_ok=True)
        except OSError as e:
            _log.warning("Failed to cleanup PDF cache file %s: %s", cached_pdf, e)

    # 清理内存缓存
    expired_hashes = [
        h for h, (_, ts) in _content_cache.items() if now - ts > CACHE_TTL_SECONDS
    ]
    for h in expired_hashes:
        del _content_cache[h]


def _compute_content_hash(content: bytes) -> str:
    """计算文件内容的SHA256哈希值"""
    return hashlib.sha256(content).hexdigest()


def _compute_url_hash(url: str) -> str:
    """计算URL的SHA256哈希值，用于PDF缓存"""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


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


def _save_to_cache(
    content_hash: str, file_id: str, markdown: str, filename: str
) -> None:
    """保存转换结果到缓存"""
    cache_path = CACHE_DIR / f"{file_id}.md"
    metadata_path = CACHE_DIR / f"{file_id}.json"

    cache_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {"name": filename, "content_hash": content_hash}, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    # 更新内存缓存
    _content_cache[content_hash] = (file_id, time.time())


def _filename_from_url(url: str) -> str:
    """从URL路径中提取PDF文件名，提取不到则返回 download.pdf"""
    try:
        path = urlparse(url).path
        name = unquote(Path(path).name)
        if name and name.lower().endswith(".pdf"):
            return name
    except Exception:
        pass
    return "download.pdf"


async def _download_pdf(url: str) -> tuple[bytes, str]:
    """
    下载 PDF 文件，优先使用 URL 缓存。
    返回 (pdf_bytes, filename)。
    """
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url_hash = _compute_url_hash(url)
    filename = _filename_from_url(url)
    cached_pdf_path = PDF_CACHE_DIR / f"{url_hash}.pdf"

    # 检查 PDF 缓存
    if cached_pdf_path.is_file():
        age = time.time() - cached_pdf_path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            _log.info(
                "pdf cache hit | url_hash=%s filename=%s", url_hash[:16], filename
            )
            return cached_pdf_path.read_bytes(), filename

    # 下载 PDF
    _log.info("downloading pdf | url=%s", url)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_PDF_DOWNLOAD_TIMEOUT),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download PDF: HTTP {e.response.status_code}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download PDF: {type(e).__name__}: {e}",
        )

    content = resp.content
    if not content:
        raise HTTPException(status_code=502, detail="Downloaded PDF is empty")

    # 简单校验 PDF magic bytes
    if not content[:5].startswith(b"%PDF-"):
        raise HTTPException(
            status_code=502,
            detail="Downloaded content is not a valid PDF",
        )

    # 写入 PDF 缓存
    cached_pdf_path.write_bytes(content)
    _log.info(
        "pdf downloaded & cached | url_hash=%s size=%.1fKB",
        url_hash[:16],
        len(content) / 1024,
    )

    return content, filename


def _resolve_pdf_url(raw_url: str) -> str:
    """
    智能解析 PDF URL。
    如果是 viewer URL（如 .../viewer.html?file=/path/to.pdf），
    自动提取 file 参数并拼接为完整的 PDF 直链。
    """
    parsed = urlparse(raw_url)
    query_params = parse_qs(parsed.query)

    file_param = query_params.get("file")
    if file_param:
        file_path = file_param[0]
        # file 参数是相对路径（如 /cypc/attachDir/...），拼接 origin
        if file_path.startswith("/"):
            return f"{parsed.scheme}://{parsed.netloc}{file_path}"
        # file 参数是完整 URL
        if file_path.startswith(("http://", "https://")):
            return file_path
        # 相对路径，基于当前目录拼接
        base_path = parsed.path.rsplit("/", 1)[0]
        return f"{parsed.scheme}://{parsed.netloc}{base_path}/{file_path}"

    return raw_url


async def _resolve_input(request: Request) -> tuple[bytes, str]:
    """
    根据 Content-Type 解析输入，返回 (pdf_bytes, filename)。
    支持 multipart/form-data（文件上传）和 application/json（URL）。
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # URL 模式
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        url = body.get("url") if isinstance(body, dict) else None
        if not url or not isinstance(url, str):
            raise HTTPException(
                status_code=400, detail='JSON body must contain "url" field'
            )

        # 基本 URL 校验
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400, detail="URL must use http or https scheme"
            )

        # 解析实际 PDF URL（支持 viewer URL）
        resolved_url = _resolve_pdf_url(url)
        if resolved_url != url:
            _log.info("resolved viewer url | %s -> %s", url, resolved_url)

        return await _download_pdf(resolved_url)

    elif "multipart/form-data" in content_type:
        # 文件上传模式
        form = await request.form()
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise HTTPException(
                status_code=400, detail='multipart/form-data must contain "file" field'
            )

        filename = getattr(file, "filename", None) or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="File must be a PDF (.pdf)")

        try:
            content = await file.read()
        except Exception as e:
            _log.warning("read error: %s", e)
            raise HTTPException(status_code=400, detail="Failed to read uploaded file")

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        return content, filename

    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be multipart/form-data (file upload) or application/json (url)",
        )


app = FastAPI(
    title="Docling PDF to Markdown",
    description="Upload PDF or provide URL, get Markdown. Uses Docling with GPU acceleration.",
    version="1.0.0",
)

# Optional: set debug logging for conversion details
_setup_file_logging()
logging.getLogger("app").setLevel(logging.DEBUG)
_log.info("File logging enabled: %s", LOG_FILE)


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
async def convert(request: Request, background_tasks: BackgroundTasks):
    """
    Convert PDF to Markdown. Accepts file upload (multipart/form-data) or URL (application/json).

    Returns a JSON list where each item contains:
    - id: generated file id
    - url: download URL
    - name: markdown filename
    - cached: whether result was from cache
    """
    background_tasks.add_task(_cleanup_expired_cache_files)
    request_start = time.perf_counter()

    content, original_filename = await _resolve_input(request)
    size_kb = len(content) / 1024
    filename = original_filename.rsplit(".", 1)[0] + ".md"

    # 计算内容哈希
    content_hash = _compute_content_hash(content)

    # 检查缓存
    cached_file_id = _check_cache(content_hash)
    if cached_file_id:
        request_duration = time.perf_counter() - request_start
        download_url = str(request.url_for("download_file", file_id=cached_file_id))
        _log.info(
            "convert cached | filename=%s size=%.1fKB hash=%s duration=%.3fs",
            filename,
            size_kb,
            content_hash[:16],
            request_duration,
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
        convert_func = partial(
            convert_pdf_to_markdown, pdf_path=tmp_path, return_summary=True
        )
        markdown, summary = await loop.run_in_executor(_executor, convert_func)
    except FileNotFoundError as e:
        _log.warning("convert failed | filename=%s: %s", original_filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        _log.warning("convert failed | filename=%s: %s", original_filename, e)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    request_duration = time.perf_counter() - request_start
    file_id = uuid4().hex

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _save_to_cache(content_hash, file_id, markdown, filename)

    download_url = str(request.url_for("download_file", file_id=file_id))

    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info(
        "convert done | filename=%s size=%.1fKB hash=%s duration=%.3fs",
        filename,
        size_kb,
        content_hash[:16],
        request_duration,
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
async def convert_report_api(request: Request, background_tasks: BackgroundTasks):
    """
    Convert financial-report PDF using report-specific preprocessing.
    Accepts file upload (multipart/form-data) or URL (application/json).
    """
    background_tasks.add_task(_cleanup_expired_cache_files)
    request_start = time.perf_counter()

    content, original_filename = await _resolve_input(request)
    size_kb = len(content) / 1024
    filename = original_filename.rsplit(".", 1)[0] + ".md"

    # 计算内容哈希（加上"report:"前缀以区分普通转换）
    content_hash = "report:" + _compute_content_hash(content)

    # 检查缓存
    cached_file_id = _check_cache(content_hash)
    if cached_file_id:
        request_duration = time.perf_counter() - request_start
        download_url = str(request.url_for("download_file", file_id=cached_file_id))
        _log.info(
            "convert_report cached | filename=%s size=%.1fKB hash=%s duration=%.3fs",
            filename,
            size_kb,
            content_hash[:23],
            request_duration,
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

    # 异步转换（保留原始文件名，convert_report 依赖文件名推断报告类型）
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / original_filename
    tmp_path.write_bytes(content)

    try:
        loop = asyncio.get_event_loop()
        convert_func = partial(convert_report, pdf_path=tmp_path, return_summary=True)
        markdown, summary = await loop.run_in_executor(_executor, convert_func)
    except FileNotFoundError as e:
        _log.warning("convert_report failed | filename=%s: %s", original_filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        _log.warning("convert_report failed | filename=%s: %s", original_filename, e)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    request_duration = time.perf_counter() - request_start
    file_id = uuid4().hex

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _save_to_cache(content_hash, file_id, markdown, filename)

    download_url = str(request.url_for("download_file", file_id=file_id))

    summary_dict = summary.to_dict()
    summary_dict["request_duration_sec"] = round(request_duration, 3)
    _log.info(
        "convert_report done | filename=%s size=%.1fKB hash=%s duration=%.3fs",
        filename,
        size_kb,
        content_hash[:23],
        request_duration,
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
