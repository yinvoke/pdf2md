#!/usr/bin/env python3
"""
Test script for resource monitoring system.
Run with: uv run python scripts/test_monitoring.py
"""
import torch
from pathlib import Path
from app.converter import convert_pdf_to_markdown


def main():
    print("=" * 70)
    print("Resource Monitoring Test")
    print("=" * 70)
    print()
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print()
    
    # Check monitoring dependencies
    try:
        import psutil
        import pynvml
        monitoring_available = True
        print("✓ Monitoring dependencies installed (psutil + pynvml)")
    except ImportError:
        monitoring_available = False
        print("✗ Monitoring dependencies not installed")
        print("  Install with: uv sync --extra monitor")
    print()
    
    # Test conversion
    test_pdf = Path("example/2025 年3季度报告.pdf")
    if not test_pdf.exists():
        print(f"Error: Test file not found: {test_pdf}")
        return 1
    
    print(f"Converting: {test_pdf.name}")
    print("-" * 70)
    
    torch.cuda.empty_cache()
    markdown, summary = convert_pdf_to_markdown(
        test_pdf,
        return_summary=True,
        verbose=True
    )
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    data = summary.to_dict()
    print(f"Pages:           {data['pages']}")
    print(f"Duration:        {data['duration_sec']:.2f}s")
    print(f"Speed:           {data['pages_per_sec']:.2f} pages/s")
    print()
    
    if monitoring_available:
        print("Resource Peaks:")
        print(f"  CPU:           {data['cpu_percent']:.1f}%")
        print(f"  RAM:           {data['ram_percent']:.1f}% ({data['ram_used_gb']:.1f}GB)")
        print(f"  GPU:           {data['gpu_util_percent']:.1f}%")
        print(f"  VRAM:          {data['vram_util_percent']:.1f}%")
    else:
        print("(Resource monitoring disabled - install with --extra monitor)")
    
    print()
    print(f"GPU Memory:")
    print(f"  Allocated:     {data['gpu_mem_allocated_mb']:.0f}MB")
    print(f"  Reserved:      {data['gpu_mem_reserved_mb']:.0f}MB")
    print(f"  Peak:          {data['gpu_mem_max_mb']:.0f}MB")
    print()
    
    print("✓ Test completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
