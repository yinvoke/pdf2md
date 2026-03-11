# pdf2md

基于 [Docling](https://github.com/docling-project/docling) 的 PDF 转 Markdown HTTP 服务，主要对财报的pdf转换进行了优化，支持 GPU（CUDA）加速。


**欢迎交流学习**

> 该项目目标主要是节约成本，目前市面上常见转换工具效果最好的是 doc2x，但是也要 「￥0.02/页」，一篇年报差不多也要 5 块以上。对于大量年报进行分析时成本较高，目前思路是对无效信息的章节进行删除处理，单篇年报/半年报大约可以节约90000字符数左右，如果有更好的思路或是更好的实现也可以提供


### Benchmark

环境：Intel 13600KF + NVIDIA RTX 3060 12GB + 64GB RAM
配置：batch=64, layout_batch=64, table_batch=32, queue=10

| 文档 | 模式 | 章节检测 | 保留/过滤 | 转换页数 | 字符数 | 耗时 | 节约页数 | 节约字符 | 节约时间 |
|------|------|---------|----------|---------|--------|------|---------|---------|---------|
| 燕京啤酒_2024年年报 (202页) | convert | — | — | 202 | 398,045 | 121.8s | — | — | — |
| | convert_report | 10 | 3/10 | 155 | 310,887 | 62.4s | **-47 (-23.3%)** | **-87,158 (-21.9%)** | **-59.4s (-48.8%)** |
| 燕京啤酒_2025年半年报 (136页) | convert | — | — | 136 | 288,665 | 60.6s | — | — | — |
| | convert_report | 9 | 3/9 | 117 | 234,842 | 51.8s | **-19 (-14.0%)** | **-53,823 (-18.6%)** | **-8.8s (-14.5%)** |
| 新和成_2024年年报 (189页) | convert | — | — | 189 | 447,026 | 134.2s | — | — | — |
| | convert_report | 10 | 3/10 | 151 | 355,821 | 68.4s | **-38 (-20.1%)** | **-91,205 (-20.4%)** | **-65.8s (-49.0%)** |
| 长江电力_2024年年报 (262页) | convert | — | — | 262 | 500,581 | 157.3s | — | — | — |
| | convert_report | 9 | 3/9 | 210 | 387,922 | 77.5s | **-52 (-19.8%)** | **-112,659 (-22.5%)** | **-79.8s (-50.7%)** |

### convert_report 说明

`convert_report` 针对年报/半年报进行了章节过滤优化，自动识别 `第X节` 章节结构，移除与财务分析无关的章节（如释义、公司简介、公司治理、股份变动、环境和社会责任等），保留核心章节（管理层讨论与分析、重要事项、财务报告），减少转换页数，从而节约时间和字符数。

> 4 篇报告合计节约 **344,845 字符 (21.1%)**。对于大量年报批量处理场景，推荐使用 `convert_report` 接口。

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

- **POST /convert**：通用 PDF 转 Markdown，`multipart/form-data` 字段 `file` 为 PDF；返回 JSON 含 `id`、`url`、`name`（结果缓存约 1 小时）
- **POST /convert_report**：财报类 PDF 转 Markdown，使用报表专用预处理；请求/返回格式同 `/convert`
- **GET /health**：`{"cuda_available": bool, "status": "ok"}`
- **GET /files/{file_id}**：按 `id` 下载转换后的 Markdown

示例：

```bash
curl -X POST "http://localhost:12138/convert" -F "file=@your.pdf"
```

返回为 JSON 数组，每个元素对应一个转换结果：

| 字段   | 说明 |
|--------|------|
| `id`   | 文件 ID，用于下载 |
| `url`  | 下载地址，即 `GET /files/{id}` 的完整 URL |
| `name` | 建议保存的 .md 文件名（由原 PDF 名推导） |

示例返回：

```json
[
  {
    "id": "a1b2c3d4e5f6...",
    "url": "http://localhost:12138/files/a1b2c3d4e5f6...",
    "name": "your.md"
  }
]
```

用返回的 `url` 或 `GET /files/{id}` 可下载 Markdown 正文。

## Docker

```bash
docker build -t pdf2md .
docker run -d --name pdf2md --gpus all -p 12138:12138 -v ./logs:/app/logs --restart unless-stopped pdf2md
```

## License

MIT. See [LICENSE](LICENSE).
