"""
在项目根目录执行:
    uv run python scripts/check_cuda.py

用途:
1) 自动检查 nvidia-smi / nvcc / torch CUDA 状态
2) 给出可直接执行的修复命令
3) 成功返回 0，失败返回非 0（便于 CI 或脚本链路判断）
"""
import subprocess
import sys
import re
from typing import Optional, Tuple


def run_cmd(args: list[str]) -> Tuple[bool, str]:
    """Run command and return (ok, output)."""
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return False, "命令不存在"
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output


def get_pkg_ver(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> int:
    print("=== PyTorch 与 CUDA 诊断 ===\n")
    print("CUDA Toolkit 下载地址: https://developer.nvidia.com/cuda-downloads\n")
    print("cuDNN 下载地址: https://developer.nvidia.com/cudnn\n")

    smi_ok, smi_out = run_cmd(["nvidia-smi"])
    nvcc_ok, nvcc_out = run_cmd(["nvcc", "--version"])
    print(f"[系统检查] nvidia-smi: {'OK' if smi_ok else '失败'}")
    if smi_out:
        print(smi_out.splitlines()[0])
    print(f"[系统检查] nvcc --version: {'OK' if nvcc_ok else '失败'}")
    if nvcc_out:
        print(nvcc_out.splitlines()[-1])
    print()

    try:
        import torch
    except Exception as exc:
        print(f"[Python 环境] 无法导入 torch: {exc}")
        print("\n建议执行：")
        print("  uv sync --extra cuda")
        print("  # 可选（某些场景需要） uv pip install nvidia-cudnn")
        return 2

    torch_ver = torch.__version__
    built = bool(torch.backends.cuda.is_built())
    avail = bool(torch.cuda.is_available())
    torch_cuda = getattr(torch.version, "cuda", None)
    try:
        cudnn_runtime_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_runtime_ver = None
    cudnn_available = bool(getattr(torch.backends.cudnn, "enabled", False))

    print("[Python 环境]")
    print(f"PyTorch version:  {torch_ver}")
    print(f"cuDNN Available:  {cudnn_available}")
    print(f"cuDNN runtime:    {cudnn_runtime_ver or '(不可用)'}")
    print(f"CUDA built in torch: {built}")
    print(f"CUDA Available:      {avail}")
    print(f"torch.version.cuda:  {torch_cuda}")
    print("\n[版本正确性验证]")
    print(f"Torch CUDA: {torch_cuda}")
    print(f"CUDA Available: {avail}")
    drv_cuda = None
    if smi_ok and smi_out:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_out)
        if match:
            drv_cuda = match.group(1)
            print(f"Driver CUDA: {drv_cuda}")
    if drv_cuda is None:
        print("nvidia-smi not found")
    elif torch_cuda:
        try:
            driver_major = int(drv_cuda.split(".")[0])
            torch_major = int(str(torch_cuda).split(".")[0])
            if driver_major >= torch_major:
                print("Version check: PASS (驱动 CUDA 主版本满足 torch 要求)")
            else:
                print("Version check: FAIL (驱动 CUDA 主版本低于 torch 要求)")
        except Exception:
            print("Version check: SKIP (版本解析失败)")

    if avail:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print("\n结论: CUDA 可用，环境正常。")
        return 0

    print("\n结论: CUDA 当前不可用。")
    if not smi_ok:
        print("- 先修复驱动环境，确保 nvidia-smi 可运行。")
    if not built:
        print("- 当前是 CPU 版 torch（或混装导致）。请重装 CUDA 版依赖：")
    else:
        print("- torch 带 CUDA 但运行时不可用。建议重装并复查驱动：")

    print("  uv pip uninstall torch")
    print("  uv sync --extra cuda")
    print("  # 可选（某些场景需要）：uv pip install nvidia-pyindex && uv pip install nvidia-cudnn")
    print("  uv run python scripts/check_cuda.py")

    return 1


if __name__ == "__main__":
    sys.exit(main())
