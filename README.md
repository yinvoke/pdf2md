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

环境：Intel 13600KF + NVIDIA RTX 3060 12GB + 64GB RAM
配置：batch=64, layout_batch=64, table_batch=32, queue=10

**章节过滤效果**（4 篇报告）：

| 文档 | 模式 | 章节 (保留/总) | 页数 (节约) | 字符数 | 耗时 | 节约字符 | 节约时间 |
|------|------|---------------|------------|--------|------|---------|---------|
| 燕京啤酒_2024年年报 | convert | — | 202 | 398,045 | 121.8s | — | — |
| | convert_report | 7/10 | 155 (-23.3%) | 310,887 | 62.4s | **-87,158 (-21.9%)** | **-59.4s (-48.8%)** |
| 燕京啤酒_2025年半年报 | convert | — | 136 | 288,665 | 60.6s | — | — |
| | convert_report | 6/9 | 117 (-14.0%) | 234,842 | 51.8s | **-53,823 (-18.6%)** | **-8.8s (-14.5%)** |
| 新和成_2024年年报 | convert | — | 189 | 447,026 | 134.2s | — | — |
| | convert_report | 7/10 | 151 (-20.1%) | 355,821 | 68.4s | **-91,205 (-20.4%)** | **-65.8s (-49.0%)** |
| 长江电力_2024年年报 | convert | — | 262 | 500,581 | 157.3s | — | — |
| | convert_report | 6/9 | 210 (-19.8%) | 387,922 | 77.5s | **-112,659 (-22.5%)** | **-79.8s (-50.7%)** |

> 4 篇报告合计节约 **344,845 字符 (21.1%)**，转换耗时平均减少 **40%+**。

**后处理效果**（9 篇财报，含年报 / 半年报 / 季报）：

| 类型 | 总字符数 | 断裂标题 | 错误密度 (每万字) | 修复后 | `<!-- image -->` |
|------|---------|---------|-----------------|-------|-----------------|
| 全量转换 (.md) | 3,098,030 | 76 | **0.25** | 0 | 19 |
| 章节过滤 (.report.md) | 2,419,714 | 67 | **0.28** | 0 | 15 |

> 单篇年报平均每万字约 0.3-0.6 处断裂标题（最严重的达 0.62/万字），后处理可将错误数降至 **0**。

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

## License

MIT. See [LICENSE](LICENSE).
