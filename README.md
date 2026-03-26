# pdf2md

基于 [Docling](https://github.com/docling-project/docling) 的 PDF 转 Markdown HTTP 服务，主要对财报的pdf转换进行了优化，支持 GPU（CUDA）加速。

**适配机型**：RTX 3060 12GB ✅ | RTX 4080 🚧 调优中

**欢迎交流学习**

> 该项目目标主要是节约成本，目前市面上常见转换工具效果最好的是 doc2x，但是也要 「￥0.02/页」，一篇年报差不多也要 5 块以上。对于大量年报进行分析时成本较高，目前思路是对无效信息的章节进行删除处理，单篇年报/半年报大约可以节约90000字符数左右，如果有更好的思路或是更好的实现也可以提供


### convert_report 说明

`convert_report` 针对年报/半年报做了两层优化：

**1. 章节过滤（转换前）**

自动识别 PDF 目录中的 `第X节` 章节结构，在转换前裁剪 PDF，移除与财务分析无关的章节（释义、公司简介、公司治理、股份变动、环境和社会责任等），仅保留核心章节（管理层讨论与分析、重要事项、财务报告），减少转换页数。

**2. Markdown 后处理（转换后）**

所有转换结果（`/convert` 和 `/convert_report`）均自动执行后处理，修复 Docling 的已知转换缺陷：

- **断裂标题修复**：Docling 常将一个完整标题拆成连续两行 `## `，导致 Markdown 目录结构错乱。后处理自动检测并合并，支持正向合并（`## 1` + `## 、企业文化优势` → `## 1、企业文化优势`）、反序合并（`## 、持续经营` + `## 2` → `## 2、持续经营`）、括号占位填充（`## （ ）权益法` + `## 2` → `## （2）权益法`）等多种模式，同时内置防误合并守卫避免将独立标题错误拼接
- **图片占位移除**：清除 Docling 生成的 `<!-- image -->` 注释行

### Benchmark

环境：Intel 13600KF + NVIDIA RTX 3060 12GB + 64GB RAM，Docling 2.82.0

- **全量转换**：11 篇（4 A 股 + 7 港股），合计 **3,200,698 字符**，乱码率 **0%**，标准模式 ~2.3-4.3 页/秒，OCR 回退模式 ~1.15-1.23 页/秒
- **章节过滤**（`convert_report`）：A 股年报/半年报平均节约 **21.3% 字符**，转换耗时减少 **~16%**
- **工具对比**：对比 kreuzberg 4.6.1，docling 在表格提取和文档结构上显著领先，kreuzberg 速度极快但无表格结构，适合纯文本场景

详见 [Benchmark 报告](benchmark/BENCHMARK.md)。

## 环境

- Python 3.10+，[uv](https://docs.astral.sh/uv/)
- 推荐cuda，没cuda太慢了，没必要
- 可选：NVIDIA GPU + CUDA（无则用 CPU）
  - CUDA 12.4+

## 安装与运行

**生产环境**（最小依赖）：

```bash
uv sync                    # 仅核心依赖
uv sync --extra cuda       # 添加 GPU 加速
```

**开发/测试环境**（含资源监控）：

```bash
uv sync --extra cuda --extra monitor
uv run python scripts/check_cuda.py
uv run python scripts/test_monitoring.py  # 测试转换和监控功能
```

> 监控功能需要 `psutil` 和 `pynvml` 依赖（`--extra monitor`），生产环境可省略以减小镜像体积。监控指标包括：CPU%、RAM GB/%、GPU利用率%、VRAM占用率%。

**启动服务：**

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 12138 --reload
```

## 接口

### POST /convert

通用 PDF 转 Markdown。支持两种输入方式（二选一）：

**方式一：文件上传**

```bash
curl -X POST "http://localhost:12138/convert" -F "file=@your.pdf"
```

**方式二：PDF 链接**

```bash
curl -X POST "http://localhost:12138/convert" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/path/to/report.pdf"}'
```

> URL 模式支持 PDF Viewer 链接（如 `viewer.html?file=/path/to.pdf`），会自动提取实际 PDF 地址。

### POST /convert_report

财报类 PDF 转 Markdown，使用报表专用预处理（章节过滤 + 后处理）。请求/返回格式同 `/convert`，同样支持文件上传和 URL 两种方式。

### GET /health

```json
{"cuda_available": true, "status": "ok"}
```

### GET /files/{file_id}

按 `id` 下载转换后的 Markdown 文件。

### 返回格式

所有转换接口返回 JSON 数组：

```json
[
  {
    "id": "a1b2c3d4e5f6...",
    "url": "http://localhost:12138/files/a1b2c3d4e5f6...",
    "name": "report.md",
    "cached": false
  }
]
```

| 字段     | 说明 |
|----------|------|
| `id`     | 文件 ID，用于下载 |
| `url`    | 下载地址，即 `GET /files/{id}` 的完整 URL |
| `name`   | 建议保存的 .md 文件名（由原 PDF 名推导） |
| `cached` | 是否命中缓存 |

### 缓存策略

| 缓存对象 | 缓存路径 | TTL | 缓存 Key |
|---------|---------|-----|---------|
| 下载的 PDF 文件 | `cache/pdf/` | 180 天 | URL 的 SHA256 |
| 生成的 Markdown | `cache/markdown/` | 180 天 | 内容 SHA256 |

> 相同 URL 或相同内容的 PDF 在缓存有效期内直接返回，不会重复下载或转换。

## Docker

```bash
docker build -t pdf2md .
docker run -d --name pdf2md --gpus all -p 12138:12138 -v ./logs:/app/logs --restart unless-stopped pdf2md
```

### GPU 透传配置

容器必须通过 NVIDIA Container Toolkit 正确透传 GPU，否则会退回 CPU 模式。CPU 模式下 `force_backend_text=True` 使用 pypdfium2 后端提取文字，对于 CID 字体（常见于港股财报等 Adobe InDesign 生成的 PDF）会产生乱码。

**前置条件：**

1. 宿主机已安装 NVIDIA 驱动（`nvidia-smi` 正常输出）
2. 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)：

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

3. 配置 Docker 默认使用 nvidia runtime：

```bash
# /etc/docker/daemon.json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

```bash
sudo systemctl restart docker
```

**验证：**

```bash
# 容器内应能看到 GPU
docker exec pdf2md nvidia-smi

# 确认 PyTorch 可用 CUDA
docker exec pdf2md .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

> ⚠️ 常见问题：`daemon.json` 中只注册了 nvidia runtime 但没有设置 `"default-runtime": "nvidia"`，导致 `--gpus all` 仍然使用 `runc`，GPU 设备无法透传（表现为容器内 `nvidia-smi` 报 "Failed to initialize NVML"、`torch.cuda.is_available()` 返回 `False`）。

### 港股财报说明

港股财报 PDF（如巨潮资讯网 `cninfo.com.cn` 的 HK 上市公司公告）与 A 股年报有以下差异：

- **无 `第X节` 章节结构**：港股财报 PDF 通常无 TOC 书签或仅有英文书签，不含 `第一节`、`第二节` 等标准分节。`convert_report` 会自动 fallback 为全量转换（等同 `convert`），无需额外处理。
- **CID 字体自动 OCR 回退**：部分港股财报使用 CID-keyed 字体（如汉仪旗黑 HYQiHei + `Identity-H` 编码），Docling 的 pypdfium2 后端无法正确提取文字。转换器会在转换前用 PyMuPDF 检测乱码字体，自动切换到 OCR 模式（`force_full_page_ocr=True`），单次转换即可得到正确结果，无需手动干预。OCR 模式速度约 1.1-1.2 页/秒（标准模式 2.3-4.3 页/秒）。
- **繁体中文 + 英文混排**：需要 GPU 模式（ML 模型识别），CPU 模式的 pypdfium2 后端对 CID 字体编码支持不完善，会产生乱码。
- **文件名通常为数字编号**（如 `1222833601.PDF`），扩展名可能为大写 `.PDF`。

## License

MIT. See [LICENSE](LICENSE).
