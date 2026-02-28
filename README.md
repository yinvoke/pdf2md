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

## 环境

- Python 3.10+，[uv](https://docs.astral.sh/uv/)
- 推荐cuda，没cuda太慢了，没必要
- 可选：NVIDIA GPU + CUDA（无则用 CPU）
  - CUDA 13.0 +
  - CUDNN (可选) 9.0+ 

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

镜像会安装 CUDA 依赖（torch/torchvision），以便 docling 的图表识别等阶段能正常导入；不传 GPU 时仍按 CPU 运行。

```bash
docker build -t docling-server:local .
docker run --rm -p 12138:12138 docling-server:local
```

使用 GPU 时需加 `--gpus all`，或使用 docker-compose：`docker-compose --profile gpu up`。

## License

MIT. See [LICENSE](LICENSE).
