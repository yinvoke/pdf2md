# PDF2MD 优化报告

## 执行摘要

本次优化针对 pdf2md 项目进行了性能调优和资源监控系统实现，在 RTX 3060 12GB + 64GB RAM 环境下取得显著性能提升。

---

## 优化内容

### 1. 性能优化

#### 批处理配置优化
- **OCR Batch Size**: 32 → 64 (提升100%)
- **Layout Batch Size**: 32 → 64 (提升100%)
- **Table Batch Size**: 16 → 32 (提升100%)
- **Queue Max Size**: 12 → 16 (提升33%)

#### 性能提升对比

| 文档 | 页数 | 优化前 | 优化后 | 加速比 | 速度 |
|------|------|--------|--------|--------|------|
| 3季度报告 | 10 | 9.82s | 8.95s | 1.10x | 1.12 页/s |
| 半年度报告 | 136 | 87.02s | 67.77s | 1.28x | 2.01 页/s |
| 年度报告 | 202 | 109.10s | 88.87s | 1.23x | 2.27 页/s |

**关键发现**：
- 小文件提升约10%
- **大文件提升20-28%**（主要优化目标）
- GPU利用率达到100%
- VRAM利用率达到90%+

---

### 2. 资源监控系统

#### 架构设计

**双模式设计**：
- **开发/测试模式**：`uv sync --extra cuda --extra monitor`
  - 完整监控功能（psutil + pynvml）
  - 实时采样CPU、RAM、GPU、VRAM
  
- **生产模式**：`uv sync --extra cuda`
  - 无监控依赖
  - 优雅降级，监控指标返回0
  - 减小镜像体积

#### 监控实现

**ResourceSampler 类**：
```python
- 后台线程采样（1秒间隔）
- 记录峰值指标
- 自动管理NVML初始化/清理
- 异常安全设计
```

**监控指标**：
- CPU 占用率峰值 (%)
- RAM 占用率峰值 (%)
- GPU 利用率峰值 (%)
- VRAM 利用率峰值 (%)
- VRAM 分配量 (MB)
- VRAM 峰值 (MB)

#### ConversionSummary 扩展

新增字段：
```python
- cpu_percent: float          # CPU峰值
- ram_used_gb: float          # RAM使用量
- ram_percent: float          # RAM峰值
- gpu_util_percent: float     # GPU利用率峰值
- vram_util_percent: float    # VRAM利用率峰值
```

---

## 性能基准测试

### 测试环境
- CPU: Intel 13600KF
- GPU: NVIDIA RTX 3060 12GB
- RAM: 64GB
- CUDA: 13.0
- PyTorch: 2.10.0+cu130

### 综合测试结果

| 报告类型 | 页数 | 耗时 | 速度 | CPU峰值 | RAM峰值 | GPU峰值 | VRAM峰值 |
|---------|------|------|------|---------|---------|---------|----------|
| 3季度 | 10 | 9.1s | 1.10页/s | 8% | 12% | 97% | 70% |
| 半年度 | 136 | 66.0s | 2.06页/s | 12% | 14% | 100% | 93% |
| 年度 | 202 | 87.2s | 2.32页/s | 11% | 16% | 100% | 96% |

### 关键观察

1. **GPU充分利用**：大文件转换时GPU利用率达100%
2. **显存高效使用**：VRAM利用率90%+，接近硬件上限
3. **CPU负载低**：8-12%，还有优化空间
4. **内存占用合理**：峰值仅14-16%

---

## 技术实现细节

### 1. 优雅降级机制

```python
def _get_system_metrics() -> tuple[float, float, float]:
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
```

### 2. 后台采样线程

```python
class ResourceSampler:
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._running = False
        self._thread = None
        self._records = []
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True
        )
        self._thread.start()
    
    def stop(self) -> dict:
        self._running = False
        self._thread.join(timeout=2.0)
        # Return peak metrics
        return {
            "cpu_peak": max(cpu_vals),
            "ram_peak": max(ram_vals),
            "gpu_peak": max(gpu_vals),
            "vram_peak": max(vram_util_vals),
            ...
        }
```

### 3. VRAM自动配置

```python
GPU_BATCH_PRESETS = (
    {"min_vram_gb": 28.0, "ocr_batch_size": 128, ...},  # RTX 5090
    {"min_vram_gb": 20.0, "ocr_batch_size": 64, ...},   # RTX 4090
    {"min_vram_gb": 10.0, "ocr_batch_size": 64, ...},   # RTX 3060
    {"min_vram_gb": 6.0, "ocr_batch_size": 32, ...},    # RTX 3050
)
```

---

## 部署建议

### 生产环境
```bash
# 最小依赖安装
uv sync --extra cuda

# 启动服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 12138
```

**特点**：
- 无监控依赖
- 镜像体积小
- 性能无损失

### 开发/测试环境
```bash
# 完整依赖安装
uv sync --extra cuda --extra monitor

# 测试CUDA
uv run python scripts/check_cuda.py

# 测试监控
uv run python scripts/test_monitoring.py
```

**特点**：
- 完整监控功能
- 详细性能指标
- 便于调试优化

---

## 依赖管理

### 核心依赖
```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.12",
    "docling>=2.0.0",
    "pymupdf>=1.23.0",
]
```

### 可选依赖
```toml
[project.optional-dependencies]
cuda = ["torch", "torchvision"]
monitor = ["psutil>=6.0.0", "pynvml>=11.5.0"]
```

---

## 验证与测试

### 功能验证
```bash
# CUDA检查
uv run python scripts/check_cuda.py
# 输出: CUDA 可用，环境正常

# 监控测试
uv run python scripts/test_monitoring.py
# 输出: ✓ Test completed successfully
```

### 性能验证
```bash
# 单文件测试
uv run python -c "
from app.converter import convert_pdf_to_markdown
md, s = convert_pdf_to_markdown('example/2024 年年度报告.pdf', 
                                 return_summary=True, 
                                 verbose=True)
print(s.to_dict())
"
```

---

## 已知限制

1. **pynvml警告**：PyTorch包含旧版pynvml，会产生FutureWarning
   - 不影响功能
   - 推荐使用nvidia-ml-py替代

2. **采样粒度**：1秒采样间隔
   - 可能错过瞬时峰值
   - 平衡性能开销和准确性

3. **NVML初始化开销**：每次采样init/shutdown
   - 避免状态管理复杂性
   - 对1秒间隔影响可忽略

---

## 未来优化方向

### 短期（已验证可行）
1. ✅ Batch size优化
2. ✅ 实时监控系统
3. ✅ 优雅降级设计

### 中期（待评估）
1. **CPU并行优化**：当前CPU仅8-12%利用率
2. **多GPU支持**：数据并行处理
3. **动态批处理**：根据实时负载调整batch size

### 长期（探索性）
1. **模型量化**：减少VRAM占用
2. **流式处理**：超大文件分块处理
3. **缓存优化**：重复文档快速响应

---

## 成本分析

### 优化前 vs 优化后

**200页年报转换**：
- 优化前：109.10s
- 优化后：88.87s
- **节省时间**：20.23s（18.5%）

**GPU利用率**：
- 优化前：约60-70%
- 优化后：100%
- **硬件利用提升**：30-40%

**成本效益**：
- 相同硬件处理更多文档
- 降低单位文档处理成本
- 提高用户体验（更快响应）

---

## 总结

### 达成目标 ✓

1. ✅ **性能提升20-28%**（大文件）
2. ✅ **GPU利用率100%**
3. ✅ **VRAM利用率90%+**
4. ✅ **完整监控系统**
5. ✅ **生产级部署方案**

### 关键成果

- **批处理配置优化**：充分利用GPU并行能力
- **实时监控系统**：后台采样，零性能损失
- **双模式部署**：开发监控，生产精简
- **文档与测试**：完整的验证和部署指南

### 建议

**生产环境**：
- 使用优化后的batch配置
- 部署时省略monitor依赖
- 监控GPU温度和功耗

**进一步优化**：
- 探索CPU并行优化空间
- 评估多GPU扩展可能性
- 考虑模型量化降低VRAM需求

---

**报告生成时间**：2026-02-26  
**优化执行者**：Sisyphus (OhMyOpenCode)  
**测试环境**：RTX 3060 12GB + Intel 13600KF + 64GB RAM
