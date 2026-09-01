# 城市交通知识与预测 Agent MVP 验收记录

- 验收日期：2026-09-01
- 功能代码提交：`ab0563e44ab9668fcc12c4fb5568649df2fe967b`
- 检索基准提交：`0e601218`
- 运行环境：实验室 Linux 服务器，CPU 推理，API `127.0.0.1:18100`，Streamlit `127.0.0.1:18501`
- GitHub 仓库：<https://github.com/zouzouqaq/traffic-knowledge-agent>
- 验收结果：MVP、可选 DeepSeek 证据回答、本地部署与自动回退验收通过

## 一、MVP 边界

本版 MVP 面向“交通领域资料检索、实验指标查询和交通流预测调用”三个场景，完成文档入库、混合检索、带引用回答、工具路由、服务降级、自动化评测和 Web 演示。系统默认使用离线证据回答，可选使用 DeepSeek 对已有证据和工具结果进行中文组织；DeepSeek 不参与工具选择，且不能添加证据。昆明实时交通数据仍不在本次边界内。

## 二、验收结果

| 序号 | 验收项 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | Markdown 文档入库 | 通过 | HTTP 上传返回 `201`，状态为 `indexed` |
| 2 | PDF 文档入库 | 通过 | HTTP 上传返回 `201`，状态为 `indexed` |
| 3 | DOCX 文档入库 | 通过 | HTTP 上传返回 `201`，状态为 `indexed` |
| 4 | 重复文档处理 | 通过 | 相同 Markdown 再次上传返回 `200`，`duplicate=true`，文档总数保持 3 |
| 5 | 混合检索与引用 | 通过 | 返回通道为 `vector+bm25`；知识回答包含可追溯引用 |
| 6 | 三工具路由 | 通过 | 依次调用 `search_traffic_knowledge`、`get_model_metrics`、`run_traffic_forecast` |
| 7 | 预测服务故障降级 | 通过 | 预测服务离线时返回 `partial=true` 与 `FORECAST_UNAVAILABLE`，知识和指标工具继续可用 |
| 8 | 50 题 Agent 固定集评测 | 通过 | 知识/指标路由准确率 `100%`，工具调用成功率 `100%`，引用正确率 `95.45%` |
| 9 | 可复现 JSON 产物 | 通过 | 评测文件包含 Git 提交、脏工作区状态、输入指纹、运行参数和指标；均可由 `json.tool` 解析 |
| 10 | 非 Docker 启动与质量检查 | 通过 | API 和 Streamlit 可直接启动；`191 passed`，Ruff 全部通过 |
| 11 | DeepSeek 证据回答 | 通过 | 10 道知识题成功率 `100%`，引用存在率 `100%`，回退率 `0%` |
| 12 | DeepSeek 故障回退 | 通过 | 无效 Key、不可达地址、1 ms 超时分别返回安全错误码并保留证据答案与引用 |

## 三、量化结果

### 3.1 检索对比

在 50 题固定回归集上：

| 检索方式 | Hit@1 | Hit@3 | MRR |
| --- | ---: | ---: | ---: |
| 向量检索 | 92% | 100% | 95.33% |
| BM25 | 82% | 98% | 90.07% |
| Hybrid | 94% | 100% | 97.00% |

Hybrid 的 Hit@1 相比 BM25 提升 12 个百分点。该数据集与系统共同设计，用于固定版本回归，不代表未知问题上的泛化能力。

### 3.2 Agent 与服务性能

- 50 题固定集：知识/指标路由准确率 100%，工具调用成功率 100%，引用正确率 95.45%。
- 确定性 CPU 请求延迟：P50 `9.58 ms`，P95 `10.46 ms`。
- 并发测试：并发数 4、请求数 30，吞吐量 `65.88 req/s`。
- 入库测试：5 篇文档、50 个分块，耗时 `6.125 s`，约 `8.16 chunks/s`，索引大小 `2,736,868 bytes`。
- 基准进程常驻内存基线约 `1.002 GiB`；本次短请求采样未观察到额外 RSS 峰值。该值不代表多进程部署总内存。

### 3.3 DeepSeek 证据回答

- 固定集前 10 道知识题：DeepSeek 成功率 `100%`，引用存在率 `100%`，回退率 `0%`。
- 生成延迟：P50 `1.22 s`，P95 `2.09 s`；该值与上面的确定性检索/工具延迟分开统计。
- Token 总量：输入 `5,612`，输出 `737`。
- 按验收时保守峰值价格估算，10 题成本约 `$0.00344`；该值是估算，不是账单。
- 本次只验证“响应成功、引用可解析和故障回退”，未使用独立 LLM judge，因此不宣称回答质量提升。

## 四、真实 HTTP 验收

验收时预测服务端口未启动，API 健康状态为 `degraded`：元数据和检索正常、预测依赖不可用。这是有意保留的故障场景，用于验证局部失败不会拖垮知识问答和指标查询。真实 DeepSeek 请求正确回答 PEMS04 包含 307 个交通节点，使用 `deepseek-v4-flash`，返回 1 条可解析引用。

故障演练使用隔离临时进程，不改写真实 `.env`：无效 Key 返回 `LLM_AUTH_FAILED`，不可达地址返回 `LLM_UNAVAILABLE`，1 ms 超时返回 `LLM_TIMEOUT`；三种情况均回退到离线证据答案。

真实 HTTP 结果保存在忽略目录：

- `artifacts/mvp_acceptance_http.json`
- `artifacts/agent_benchmark_final.json`
- `artifacts/agent_benchmark_deepseek_acceptance.json`
- `artifacts/deepseek_answer_metrics.json`
- `artifacts/retrieval_metrics.json`

产物不提交 Git，避免将运行数据和可能持续增长的索引放入源码仓库。

## 五、安全与仓库检查

- 已检查 Git 跟踪文件，未发现常见密钥、令牌或密码模式。
- 未跟踪 `.pt`、`.pth`、`.ckpt`、`.bin`、`.onnx`、PDF 或 DOCX 大文件。
- 数据库、索引、上传文件和评测产物由 `.gitignore` 排除。
- DeepSeek Key 仅保存在权限为 `600` 的忽略文件 `.env`；评测产物不保存问题、回答、提示词、证据正文或请求头。
- 验收期间未使用 GPU，未执行 `systemctl`，未重启或清理共享 Docker，也未访问其他用户目录。

## 六、复现命令

```bash
cd /8t/usr/zhouh2024/projects/traffic-knowledge-agent/.worktrees/mvp

.venv/bin/python -m pytest -p no:cacheprovider -q
.venv/bin/python -m ruff check --no-cache .
.venv/bin/python -m json.tool artifacts/retrieval_metrics.json >/dev/null
.venv/bin/python -m json.tool artifacts/agent_benchmark_deepseek_acceptance.json >/dev/null
.venv/bin/python -m json.tool artifacts/deepseek_answer_metrics.json >/dev/null
.venv/bin/python -m json.tool artifacts/mvp_acceptance_http.json >/dev/null
```

## 七、发布状态

仓库已公开发布，默认分支为 `main`；服务器的 `feature/mvp` 同时跟踪远端同名分支。运行数据、索引和模型文件继续保留在服务器忽略目录中，不进入公开仓库。
