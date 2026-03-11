# pdf2md

基于 [Docling](https://github.com/docling-project/docling) 的 PDF 转 Markdown HTTP 服务，主要对财报的pdf转换进行了优化，支持 GPU（CUDA）加速。


**欢迎交流学习**

> 该项目目标主要是节约成本，目前市面上常见转换工具效果最好的是 doc2x，但是也要 「￥0.02/页」，一篇年报差不多也要 5 块以上。对于大量年报进行分析时成本较高，目前思路是对无效信息的章节进行删除处理，单篇年报/半年报大约可以节约90000字符数左右，如果有更好的思路或是更好的实现也可以提供


### Benchmark

环境：Intel 13600KF + NVIDIA RTX 3060 12GB + 64GB RAM
配置：batch=64, layout_batch=64, table_batch=32, queue=10

| 文档 | 页数 | 字符数 | 耗时 | 速度 | CPU峰值 | RAM峰值 | GPU峰值 | VRAM峰值 |
|------|------|--------|------|------|---------|---------|---------|----------|
| 3季度报告 | 10 | 30,097 | 9.1s | 1.10页/s | 8% | 12% | 97% | 70% |
| 半年度报告 | 136 | 288,668 | 66.0s | 2.06页/s | 12% | 14% | 100% | 93% |
| 年度报告 | 202 | 398,045 | 87.2s | 2.32页/s | 11% | 16% | 100% | 96% |

> 优化后性能提升：batch size翻倍，大文件速度提升20-28%，GPU利用率100%，显存利用率90%+

### convert vs convert_report 对比

`convert_report` 针对年报/半年报进行了章节过滤优化，自动移除与财务分析无关的章节（如公司简介、公司治理、环境和社会责任等），减少转换页数，从而节约时间和字符数。季度报告因篇幅较短不做过滤，直接走全量转换。

| 文档 | 模式 | 转换页数 | 字符数 | 耗时 | 节约字符 | 节约时间 |
|------|------|----------|--------|------|----------|----------|
| 年度报告 (202页) | convert | 202 | 398,045 | 77.8s | — | — |
| | convert_report | 158 | 327,947 | 63.5s | **-70,098 (-17.6%)** | **-14.2s (-18.3%)** |
| 半年度报告 (136页) | convert | 136 | 288,665 | 57.4s | — | — |
| | convert_report | 90 | 226,469 | 51.2s | **-62,196 (-21.5%)** | **-6.2s (-10.8%)** |
| 3季度报告 (10页) | convert | 10 | 30,097 | 7.0s | — | — |
| | convert_report | 10 | 30,097 | 7.0s | 无过滤 (自动 fallback) | 无过滤 |

> 年报 + 半年报合计节约 **132,294 字符 (18.5%)**、**20.4 秒 (14.4%)**。对于大量年报批量处理场景，推荐使用 `convert_report` 接口。

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
