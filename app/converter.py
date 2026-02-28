"""
Docling PDF to Markdown converter with GPU acceleration and debug/summary logging.
"""
from __future__ import annotations

import os
import logging
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple

# On Windows non-admin terminals, HuggingFace symlink cache may fail with WinError 1314.
# Disable symlink usage by default to make first-time model downloads robust.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

_log = logging.getLogger(__name__)


def _verbose_print(verbose: bool, msg: str, *args) -> None:
    """Print formatted debug details only when verbose is enabled."""
    if not verbose:
        return
    print(msg % args if args else msg)

# Pipeline tuning knobs (adjust here for local GPU/CPU benchmark tests).
# GPU: 提高 batch 和 queue 以充分利用 GPU，减少空转。
# CPU: 保守配置，避免大 PDF 在 Windows 上 OOM。
GPU_PIPELINE_CONFIG = {
    "ocr_batch_size": 16,
    "layout_batch_size": 16,
    "table_batch_size": 8,
    "queue_max_size": 8,
    "do_ocr": False,
    "do_table_structure": True,
    "images_scale": 1.0,
    "force_backend_text": False,
}

CPU_PIPELINE_CONFIG = {
    "ocr_batch_size": 1,
    "layout_batch_size": 1,
    "table_batch_size": 1,
    "queue_max_size": 2,
    "do_ocr": False,
    "do_table_structure": False,
    "images_scale": 0.5,
    "force_backend_text": True,
}

# Auto batch presets by GPU VRAM (GB). 更高 VRAM = 更大 batch + 更大 queue，减少 pipeline 空转。
# Docling 官方建议: RTX 5090 64-128, RTX 4090 32-64, RTX 5070 16-32
# Queue size 适当调小以防止内存溢出
GPU_BATCH_PRESETS = (
    {
        "min_vram_gb": 28.0,
        "ocr_batch_size": 128,
        "layout_batch_size": 128,
        "table_batch_size": 64,
        "queue_max_size": 16,
    },
    {
        "min_vram_gb": 20.0,
        "ocr_batch_size": 64,
        "layout_batch_size": 64,
        "table_batch_size": 32,
        "queue_max_size": 12,
    },
    {
        "min_vram_gb": 10.0,
        "ocr_batch_size": 64,
        "layout_batch_size": 64,
        "table_batch_size": 32,
        "queue_max_size": 10,
    },
    {
        "min_vram_gb": 6.0,
        "ocr_batch_size": 32,
        "layout_batch_size": 32,
        "table_batch_size": 16,
        "queue_max_size": 8,
    },
)


def _get_gpu_memory_mb() -> tuple[float, float]:
    """Return (allocated_mb, reserved_mb). If CUDA not available, return (0, 0)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / (1024 * 1024),
            torch.cuda.memory_reserved() / (1024 * 1024),
        )
    except Exception:
        return 0.0, 0.0


def _get_max_gpu_memory_mb() -> float:
    """Return max memory allocated in MB, or 0 if not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _get_gpu_total_memory_gb() -> float:
    """Return CUDA device 0 total memory in GB, or 0 if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _get_system_metrics() -> tuple[float, float, float]:
    """
    Return (cpu_percent, ram_gb, ram_percent).
    Requires psutil (install with: uv sync --extra monitor).
    Returns (0, 0, 0) if psutil not available.
    """
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return cpu, mem.used / (1024 ** 3), mem.percent
    except ImportError:
        # psutil not installed - monitoring disabled
        return 0.0, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0


def _get_gpu_utilization() -> tuple[float, float]:
    """
    Return (gpu_util_percent, vram_util_percent).
    Requires pynvml (install with: uv sync --extra monitor).
    Returns (0, 0) if pynvml not available.
    """
    try:
        import pynvml
        # Initialize and shutdown in each call to avoid state management issues
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_percent = (mem_info.used / mem_info.total) * 100
            return float(util.gpu), vram_percent
        finally:
            pynvml.nvmlShutdown()
    except ImportError:
        # pynvml not installed - monitoring disabled
        return 0.0, 0.0


class ResourceSampler:
    """
    Background resource sampler for monitoring peak utilization during conversion.
    Requires psutil + pynvml (install with: uv sync --extra monitor).
    """
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._records: List[Tuple[float, float, float, float, float, float]] = []
        self._gpu_handle = None
        self._nvml_available = False
        
        # Try to initialize NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
        except Exception:
            pass
    
    def _sample_once(self) -> Tuple[float, float, float, float, float, float]:
        """Return (timestamp, cpu%, ram%, gpu%, vram%, vram_mb)."""
        timestamp = time.time()
        cpu = ram = gpu_util = vram_util = vram_mb = 0.0
        
        # CPU & RAM
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except (ImportError, Exception):
            pass
        
        # GPU & VRAM
        if self._nvml_available and self._gpu_handle:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(util.gpu)
                vram_util = float(util.memory)
                
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_mb = meminfo.used / (1024 ** 2)
            except Exception:
                pass
        
        return (timestamp, cpu, ram, gpu_util, vram_util, vram_mb)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            self._records.append(self._sample_once())
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start background monitoring."""
        if self._running:
            return
        self._records.clear()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> dict:
        """
        Stop monitoring and return peak metrics.
        Returns: {cpu_peak%, ram_peak%, gpu_peak%, vram_peak%, vram_peak_mb, sample_count}
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Cleanup NVML
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        if not self._records:
            return {
                "cpu_peak": 0.0,
                "ram_peak": 0.0,
                "gpu_peak": 0.0,
                "vram_peak": 0.0,
                "vram_peak_mb": 0.0,
                "sample_count": 0,
            }
        
        # Calculate peak values
        cpu_vals = [r[1] for r in self._records]
        ram_vals = [r[2] for r in self._records]
        gpu_vals = [r[3] for r in self._records]
        vram_util_vals = [r[4] for r in self._records]
        vram_mb_vals = [r[5] for r in self._records]
        
        return {
            "cpu_peak": max(cpu_vals),
            "ram_peak": max(ram_vals),
            "gpu_peak": max(gpu_vals),
            "vram_peak": max(vram_util_vals),
            "vram_peak_mb": max(vram_mb_vals),
            "sample_count": len(self._records),
        }

def _build_gpu_pipeline_config(*, verbose: bool = False) -> dict:
    """
    Build GPU config using VRAM-based presets.

    Priority:
    1) Env var DOCLING_GPU_BATCH_SIZE (apply to OCR/Layout, table uses half)
    2) VRAM presets (32/24/12GB tiers)
    3) Base GPU_PIPELINE_CONFIG defaults
    """
    cfg = dict(GPU_PIPELINE_CONFIG)

    env_batch = os.getenv("DOCLING_GPU_BATCH_SIZE", "").strip()
    if env_batch:
        try:
            batch = max(1, int(env_batch))
            cfg["ocr_batch_size"] = batch
            cfg["layout_batch_size"] = batch
            cfg["table_batch_size"] = max(1, batch // 2)
            _verbose_print(
                verbose,
                "GPU config source=env DOCLING_GPU_BATCH_SIZE=%s (ocr/layout=%s, table=%s)",
                env_batch,
                cfg["ocr_batch_size"],
                cfg["table_batch_size"],
            )
            return cfg
        except ValueError:
            _log.warning("Invalid DOCLING_GPU_BATCH_SIZE=%s, fallback to VRAM presets.", env_batch)

    total_gb = _get_gpu_total_memory_gb()
    selected_preset: Optional[dict] = None
    for preset in GPU_BATCH_PRESETS:
        if total_gb >= preset["min_vram_gb"]:
            cfg["ocr_batch_size"] = preset["ocr_batch_size"]
            cfg["layout_batch_size"] = preset["layout_batch_size"]
            cfg["table_batch_size"] = preset["table_batch_size"]
            cfg["queue_max_size"] = preset["queue_max_size"]
            selected_preset = preset
            break

    _verbose_print(
        verbose,
        "GPU config source=%s vram_gb=%.1f, ocr_batch=%s, layout_batch=%s, table_batch=%s, queue_max=%s",
        (
            f"GPU_BATCH_PRESETS(min_vram_gb>={selected_preset['min_vram_gb']})"
            if selected_preset is not None
            else "GPU_PIPELINE_CONFIG(default)"
        ),
        total_gb,
        cfg["ocr_batch_size"],
        cfg["layout_batch_size"],
        cfg["table_batch_size"],
        cfg["queue_max_size"],
    )
    return cfg


class ConversionSummary:
    """Summary of a single conversion for logging and API response."""

    def __init__(
        self,
        *,
        pages: int = 0,
        duration_sec: float = 0.0,
        pages_per_sec: float = 0.0,
        cpu_percent: float = 0.0,
        ram_used_gb: float = 0.0,
        ram_percent: float = 0.0,
        gpu_util_percent: float = 0.0,
        gpu_mem_allocated_mb: float = 0.0,
        gpu_mem_reserved_mb: float = 0.0,
        gpu_mem_max_mb: float = 0.0,
        vram_util_percent: float = 0.0,
        success: bool = False,
        error: Optional[str] = None,
    ):
        self.pages = pages
        self.duration_sec = duration_sec
        self.pages_per_sec = pages_per_sec
        self.cpu_percent = cpu_percent
        self.ram_used_gb = ram_used_gb
        self.ram_percent = ram_percent
        self.gpu_util_percent = gpu_util_percent
        self.gpu_mem_allocated_mb = gpu_mem_allocated_mb
        self.gpu_mem_reserved_mb = gpu_mem_reserved_mb
        self.gpu_mem_max_mb = gpu_mem_max_mb
        self.vram_util_percent = vram_util_percent
        self.success = success
        self.error = error
        self.pages = pages
        self.duration_sec = duration_sec
        self.pages_per_sec = pages_per_sec
        self.gpu_mem_allocated_mb = gpu_mem_allocated_mb
        self.gpu_mem_reserved_mb = gpu_mem_reserved_mb
        self.gpu_mem_max_mb = gpu_mem_max_mb
        self.success = success
        self.error = error

    def to_log_line(self) -> str:
        return (
            f"[CONVERT SUMMARY] pages={self.pages}, duration_sec={self.duration_sec:.3f}, "
            f"pages_per_sec={self.pages_per_sec:.2f}, cpu={self.cpu_percent:.1f}%, "
            f"ram={self.ram_used_gb:.1f}GB({self.ram_percent:.1f}%), "
            f"gpu_util={self.gpu_util_percent:.1f}%, vram_util={self.vram_util_percent:.1f}%, "
            f"gpu_mem_allocated_mb={self.gpu_mem_allocated_mb:.1f}, "
            f"gpu_mem_reserved_mb={self.gpu_mem_reserved_mb:.1f}, gpu_mem_max_mb={self.gpu_mem_max_mb:.1f}, "
            f"success={self.success}"
        )

    def to_dict(self) -> dict:
        return {
            "pages": self.pages,
            "duration_sec": round(self.duration_sec, 3),
            "pages_per_sec": round(self.pages_per_sec, 2),
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_gb": round(self.ram_used_gb, 1),
            "ram_percent": round(self.ram_percent, 1),
            "gpu_util_percent": round(self.gpu_util_percent, 1),
            "gpu_mem_allocated_mb": round(self.gpu_mem_allocated_mb, 1),
            "gpu_mem_reserved_mb": round(self.gpu_mem_reserved_mb, 1),
            "gpu_mem_max_mb": round(self.gpu_mem_max_mb, 1),
            "vram_util_percent": round(self.vram_util_percent, 1),
            "success": self.success,
            "error": self.error,
        }


# Module-level singleton converter (lazy init)
_converter: Optional[DocumentConverter] = None
REPORT_REMOVE_KEYWORDS = (
    "目录和释义",
    "公司简介",
    "主要财务指标",
    "公司治理",
    "环境和社会",
    "其他报送数据",
)


def _cuda_available() -> bool:
    """Return whether CUDA is actually available for this runtime."""
    try:
        import torch
        return torch.backends.cuda.is_built() and torch.cuda.is_available()
    except Exception:
        return False


def get_accelerator_device() -> str:
    """Return 'cuda' or 'cpu' for current runtime. Use for test/benchmark logging."""
    return "cuda" if _cuda_available() else "cpu"


def _get_converter(*, verbose: bool = False) -> DocumentConverter:
    global _converter
    if _converter is None:
        init_start = time.perf_counter()
        cuda_ok = _cuda_available()
        pipeline_cfg = _build_gpu_pipeline_config(verbose=verbose) if cuda_ok else CPU_PIPELINE_CONFIG
        accelerator_device = (
            AcceleratorDevice.CUDA if cuda_ok else AcceleratorDevice.CPU
        )

        # 强制 GPU 路径：设置 DOCLING_DEVICE 避免子模块从 env 读到 auto/cpu
        if cuda_ok:
            os.environ["DOCLING_DEVICE"] = "cuda"
            # page_batch_size 需 >= layout_batch_size 才能启用 layout 的 GPU 批推理
            settings.perf.page_batch_size = max(
                settings.perf.page_batch_size,
                pipeline_cfg["layout_batch_size"],
            )

        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=accelerator_device),
            ocr_batch_size=pipeline_cfg["ocr_batch_size"],
            layout_batch_size=pipeline_cfg["layout_batch_size"],
            table_batch_size=pipeline_cfg["table_batch_size"],
            queue_max_size=pipeline_cfg["queue_max_size"],
        )
        pipeline_options.do_ocr = pipeline_cfg["do_ocr"]
        pipeline_options.do_table_structure = pipeline_cfg["do_table_structure"]
        pipeline_options.images_scale = pipeline_cfg["images_scale"]
        pipeline_options.force_backend_text = pipeline_cfg["force_backend_text"]
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
        _converter.initialize_pipeline(InputFormat.PDF)
        _verbose_print(
            verbose,
            "Pipeline initialized. device=%s, ocr_batch=%s, layout_batch=%s, table_batch=%s, queue_max=%s, do_ocr=%s, do_table_structure=%s, images_scale=%.2f, force_backend_text=%s, init_duration_sec=%.3f",
            accelerator_device.value,
            pipeline_cfg["ocr_batch_size"],
            pipeline_cfg["layout_batch_size"],
            pipeline_cfg["table_batch_size"],
            pipeline_cfg["queue_max_size"],
            pipeline_cfg["do_ocr"],
            pipeline_cfg["do_table_structure"],
            pipeline_cfg["images_scale"],
            pipeline_cfg["force_backend_text"],
            time.perf_counter() - init_start,
        )
    else:
        _verbose_print(verbose, "Pipeline reused from singleton cache.")
    return _converter


def convert_pdf_to_markdown(
    pdf_path: Path | str,
    *,
    return_summary: bool = False,
    verbose: bool = False,
) -> str | tuple[str, ConversionSummary]:
    """
    Convert a PDF file to Markdown using Docling with GPU acceleration.

    :param pdf_path: Path to the PDF file.
    :param return_summary: If True, return (markdown, summary); else return markdown only.
    :param verbose: If True, emit detailed conversion logs.
    :return: Markdown string, or (markdown, summary) if return_summary is True.
    :raises ValueError: If conversion fails.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    _verbose_print(verbose, "[STEP 1/3] Input checked: path=%s, size_mb=%.2f", path, file_size_mb)

    converter_start = time.perf_counter()
    conv = _get_converter(verbose=verbose)
    converter_duration = time.perf_counter() - converter_start
    _verbose_print(verbose, "[STEP 2/3] Converter ready: duration_sec=%.3f", converter_duration)
    summary = ConversionSummary()

    # Initialize CPU monitoring (call once to prime psutil cache if available)
    try:
        import psutil
        psutil.cpu_percent(interval=None)
    except ImportError:
        pass  # psutil not installed, monitoring will be disabled
    
    # Start background resource monitoring (1 sample per second)
    sampler = ResourceSampler(sample_interval=1.0)
    sampler.start()
    
    cpu_before, ram_before_gb, ram_before_pct = _get_system_metrics()
    mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
    gpu_util_before, vram_util_before = _get_gpu_utilization()
    
    start = time.perf_counter()
    _verbose_print(
        verbose,
        "[STEP 3/3] Conversion started (monitoring every 1.0s): cpu=%.1f%%, ram=%.1fGB(%.1f%%), gpu_util=%.1f%%, vram_util=%.1f%%, gpu_mem_alloc=%.1fMB, gpu_mem_reserved=%.1fMB",
        cpu_before,
        ram_before_gb,
        ram_before_pct,
        gpu_util_before,
        vram_util_before,
        mem_before_alloc,
        mem_before_reserved,
    )

    try:
        result = conv.convert(path)
        duration = time.perf_counter() - start
        
        # Stop monitoring and get peak metrics
        peak_metrics = sampler.stop()
        
        # Sample metrics immediately after conversion
        cpu_after, ram_after_gb, ram_after_pct = _get_system_metrics()
        gpu_util_after, vram_util_after = _get_gpu_utilization()
        mem_alloc, mem_reserved = _get_gpu_memory_mb()
        max_mem = _get_max_gpu_memory_mb()

        if result.status not in {
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        }:
            summary.success = False
            summary.error = str(result.status)
            summary.duration_sec = duration
            summary.gpu_mem_allocated_mb = mem_alloc
            summary.gpu_mem_reserved_mb = mem_reserved
            summary.gpu_mem_max_mb = max_mem
            _log.warning("Conversion failed: status=%s", result.status)
            if verbose:
                _verbose_print(verbose, summary.to_log_line())
                _verbose_print(
                    verbose,
                    "Conversion details: status=%s, gpu_before_alloc_mb=%.1f, gpu_before_reserved_mb=%.1f, summary=%s",
                    result.status,
                    mem_before_alloc,
                    mem_before_reserved,
                    summary.to_dict(),
                )
            raise ValueError(f"Conversion failed: {result.status}")

        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.warning(
                "Conversion partial success: some pages failed but markdown will still be exported."
            )

        num_pages = len(result.pages) if result.pages else 0
        summary.pages = num_pages
        summary.duration_sec = duration
        summary.pages_per_sec = num_pages / duration if duration > 0 else 0.0
        # Use peak metrics from continuous sampling
        summary.cpu_percent = peak_metrics["cpu_peak"]
        summary.ram_used_gb = ram_after_gb  # Use final RAM as representative
        summary.ram_percent = peak_metrics["ram_peak"]
        summary.gpu_util_percent = peak_metrics["gpu_peak"]
        summary.gpu_mem_allocated_mb = mem_alloc
        summary.gpu_mem_reserved_mb = mem_reserved
        summary.gpu_mem_max_mb = max_mem
        summary.vram_util_percent = peak_metrics["vram_peak"]
        summary.success = True

        _verbose_print(
            verbose,
            "Conversion finished: duration_sec=%.3f, pages=%s, pages_per_sec=%.2f, samples=%d, "
            "peak: cpu=%.1f%%, ram=%.1f%%, gpu=%.1f%%, vram=%.1f%%, "
            "gpu_alloc_mb=%.1f, gpu_reserved_mb=%.1f, gpu_max_mb=%.1f",
            duration,
            num_pages,
            summary.pages_per_sec,
            peak_metrics["sample_count"],
            peak_metrics["cpu_peak"],
            peak_metrics["ram_peak"],
            peak_metrics["gpu_peak"],
            peak_metrics["vram_peak"],
            mem_alloc,
            mem_reserved,
            max_mem,
        )
        _verbose_print(verbose, summary.to_log_line())
        _verbose_print(
            verbose,
            "Conversion details: status=%s, gpu_before_alloc_mb=%.1f, gpu_before_reserved_mb=%.1f, summary=%s",
            result.status,
            mem_before_alloc,
            mem_before_reserved,
            summary.to_dict(),
        )

        markdown = result.document.export_to_markdown()
        if return_summary:
            return markdown, summary
        return markdown

    except Exception as e:
        duration = time.perf_counter() - start
        
        # Stop monitoring and get peak metrics
        peak_metrics = sampler.stop()
        
        # Sample metrics on error
        cpu_after, ram_after_gb, ram_after_pct = _get_system_metrics()
        gpu_util_after, vram_util_after = _get_gpu_utilization()
        mem_alloc, mem_reserved = _get_gpu_memory_mb()
        
        summary.success = False
        summary.error = str(e)
        summary.duration_sec = duration
        summary.cpu_percent = peak_metrics["cpu_peak"]
        summary.ram_used_gb = ram_after_gb
        summary.ram_percent = peak_metrics["ram_peak"]
        summary.gpu_util_percent = peak_metrics["gpu_peak"]
        summary.gpu_mem_allocated_mb = mem_alloc
        summary.gpu_mem_reserved_mb = mem_reserved
        summary.gpu_mem_max_mb = _get_max_gpu_memory_mb()
        summary.vram_util_percent = peak_metrics["vram_peak"]
        _log.exception("Conversion error: %s", e)
        if verbose:
            _verbose_print(verbose, summary.to_log_line())
            _verbose_print(
                verbose,
                "Conversion details: gpu_before_alloc_mb=%.1f, gpu_before_reserved_mb=%.1f, summary=%s",
                mem_before_alloc,
                mem_before_reserved,
                summary.to_dict(),
            )
        raise


def _infer_report_type_from_filename(filename: str) -> str:
    """Infer report type from file name."""
    stem = Path(filename).stem
    if "半年度报告" in stem:
        return "semi-annual"
    if "年度报告" in stem and "半年度" not in stem:
        return "annual"
    return "other"


def _extract_report_chapters_from_toc(pdf_path: Path) -> list[dict]:
    """
    Extract chapter ranges from PDF TOC.
    Return: [{title, start_page, end_page}] where page is 1-based inclusive.
    """
    import pymupdf  # type: ignore

    doc = pymupdf.open(str(pdf_path))
    try:
        toc = doc.get_toc(simple=True) or []
        if not toc:
            return []

        normalized: list[tuple[int, str, int]] = []
        for item in toc:
            if len(item) < 3:
                continue
            level, title, page = item[0], item[1], item[2]
            if not isinstance(level, int) or not isinstance(page, int):
                continue
            title_text = str(title).strip()
            if not title_text or page < 1:
                continue
            normalized.append((level, title_text, page))

        if not normalized:
            return []

        top_level = min(level for level, _, _ in normalized)
        top_nodes = [(title, page) for level, title, page in normalized if level == top_level]
        if len(top_nodes) < 2:
            return []

        dedup_nodes: list[tuple[str, int]] = []
        seen_pages: set[int] = set()
        for title, page in sorted(top_nodes, key=lambda x: x[1]):
            if page in seen_pages:
                continue
            seen_pages.add(page)
            dedup_nodes.append((title, page))

        if len(dedup_nodes) < 2:
            return []

        total_pages = doc.page_count
        chapters: list[dict] = []
        for idx, (title, start_page) in enumerate(dedup_nodes):
            end_page = dedup_nodes[idx + 1][1] - 1 if idx < len(dedup_nodes) - 1 else total_pages
            start_page = max(1, min(start_page, total_pages))
            end_page = max(1, min(end_page, total_pages))
            if end_page < start_page:
                continue
            chapters.append(
                {
                    "title": title,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )
        return chapters
    finally:
        doc.close()


def _filter_report_chapters(chapters: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop chapters by keyword and one-page rule."""
    kept: list[dict] = []
    removed: list[dict] = []

    for chapter in chapters:
        title = str(chapter.get("title", "")).strip()
        start_page = int(chapter["start_page"])
        end_page = int(chapter["end_page"])
        page_count = end_page - start_page + 1

        hit_keyword = next((kw for kw in REPORT_REMOVE_KEYWORDS if kw in title), None)
        if hit_keyword is not None:
            removed.append({**chapter, "reason": f"keyword:{hit_keyword}"})
            continue

        if page_count <= 1:
            removed.append({**chapter, "reason": "single_page"})
            continue

        kept.append(chapter)

    return kept, removed


def _build_filtered_report_pdf(source_pdf: Path, output_pdf: Path, chapters: list[dict]) -> None:
    """Merge selected chapter ranges into one temporary report PDF."""
    import pymupdf  # type: ignore

    if not chapters:
        raise ValueError("No chapter pages left after filtering.")

    source_doc = pymupdf.open(str(source_pdf))
    try:
        merged_doc = pymupdf.open()
        try:
            # Always keep the original cover page to preserve report title/context.
            if source_doc.page_count > 0:
                merged_doc.insert_pdf(source_doc, from_page=0, to_page=0)

            for chapter in chapters:
                start_page = int(chapter["start_page"])
                end_page = int(chapter["end_page"])
                # Avoid duplicating the first page when the first chapter already covers it.
                if start_page <= 1:
                    start_page = 2
                if end_page < start_page:
                    continue

                merged_doc.insert_pdf(
                    source_doc,
                    from_page=start_page - 1,
                    to_page=end_page - 1,
                )
            merged_doc.save(str(output_pdf))
        finally:
            merged_doc.close()
    finally:
        source_doc.close()


def convert_report(
    pdf_path: Path | str,
    *,
    return_summary: bool = False,
    verbose: bool = False,
) -> str | tuple[str, ConversionSummary]:
    """
    Convert report PDF with TOC-based filtering for annual/semi-annual reports.
    Non annual/semi-annual files fall back to full conversion.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    report_type = _infer_report_type_from_filename(path.name)
    if report_type not in {"annual", "semi-annual"}:
        _verbose_print(verbose, "Report mode: fallback to full conversion for non annual/semi-annual file.")
        return convert_pdf_to_markdown(path, return_summary=return_summary, verbose=verbose)

    try:
        chapters = _extract_report_chapters_from_toc(path)
    except Exception as e:
        _log.warning("Report mode TOC extraction failed, fallback to full conversion: %s", e)
        return convert_pdf_to_markdown(path, return_summary=return_summary, verbose=verbose)

    if not chapters:
        _verbose_print(verbose, "Report mode: TOC unavailable, fallback to full conversion.")
        return convert_pdf_to_markdown(path, return_summary=return_summary, verbose=verbose)

    kept_chapters, removed_chapters = _filter_report_chapters(chapters)
    if not kept_chapters:
        _verbose_print(
            verbose,
            "Report mode: all chapters removed (removed=%s), fallback to full conversion.",
            len(removed_chapters),
        )
        return convert_pdf_to_markdown(path, return_summary=return_summary, verbose=verbose)

    tmp_pdf_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf_path = Path(tmp_pdf.name)

        _build_filtered_report_pdf(path, tmp_pdf_path, kept_chapters)
        _verbose_print(
            verbose,
            "Report mode: source_chapters=%s, kept=%s, removed=%s, filtered_pdf=%s",
            len(chapters),
            len(kept_chapters),
            len(removed_chapters),
            tmp_pdf_path,
        )
        return convert_pdf_to_markdown(tmp_pdf_path, return_summary=return_summary, verbose=verbose)
    except Exception as e:
        _log.warning("Report mode failed, fallback to full conversion: %s", e)
        return convert_pdf_to_markdown(path, return_summary=return_summary, verbose=verbose)
    finally:
        if tmp_pdf_path is not None:
            tmp_pdf_path.unlink(missing_ok=True)
