VERDICT = EXEC-VERIFIED-PARTIAL

# Cycle-3 Execution-Verified First-Divergence v2

本轮仅在无 GPU 远端重用已保存的 Dynamic-v1/Cycle-3 轨迹和可执行环境；没有新模型推理、训练、搜索或数据生成。原始 FD、terminal-stop 和上一轮 semantic audit 均未修改。

## 回归门

125/125 条已审计候选复现，分歧 0；其中已知的 123 条合法并行仍判为合法，另外 2 条未被误判为合法。新旧回归缓存哈希一致。

## 新旧对比

| 指标 | 旧版 | Execution-verified v2 |
|---|---:|---:|
| FD 总数 | 1,936 | 998 |
| 保守可用 FD | 918 | 989 |
| 第 1 步 FD | 1,734 | 796 |
| 第 2 步 FD | 192 | 192 |
| 第 3 步 FD | 8 | 8 |
| 第 4 步及以后 | 2 | 2 |
| 深度 ≥2 | 202 | 202 |
| WRONG_ARGUMENT | 407 | 413 |
| semantic audit 未判定 multi-call | 895 | 615 |

新版平均/中位 FD gold step 为 1.215/1。413 条 WRONG_ARGUMENT 中，398 条来自未变的原有 direct 单调用路径，15 条由 multi-call 重新归类。9 条 `finish_reason=length` 的预算截断仍在 FD taxonomy 中，但 `keep_for_training=false`，因此 998 条 FD 中只有 989 条被标为保守可用。

## 多调用判定

1,040 个首个 FD 为多调用的候选：314 个 `VALID_PARALLEL`，9 个固定 gold 后缀验证通过的 `VALID_ALTERNATIVE`；其中 165 个包含额外调用，联合执行后的 state 与固定 gold 后缀均通过，但未逐一证明额外调用本身是 read-only。另有 1 个有实际未来 observation 证据的依赖违例，15 个错参，5 个错工具，81 个执行失败，615 个仍不确定。未判定者没有进入 FD 训练候选。

上一轮 895 个未判定候选中，本轮证实合法并行 191 个、固定后缀合法 alternative 9 个；定位执行失败 72 个、错参 3 个、错工具 5 个；仍不确定 615 个。没有将无法证明的任意替代工具路径强行归为失败。

## Frontier 迁移与重要限制

新暴露的更深层 FD = 0；其中新错参 = 0、新错工具 = 0。原因不是证明不存在更深错误，而是这批 Cycle-3 collector 遇到 multi-call 即标为 invalid 并终止：已落盘的 1,043 个多调用轮次中，后面保存的学生轮次为 0。新版已经实现并用 CPU 单测验证 gold pointer re-alignment，但不能在禁止新模型推理的前提下从不存在的后续轨迹中恢复更深错误。无 GPU 远端也不能用于重新 rollout。

这与已检查的 verl/SGLang AgentLoop `asyncio.gather` 多工具执行语义要区分：本轮 counterfactual replay 按该语义执行，而旧 collector 当时没有执行多调用。只用 shared state reset、已验证 gold prefix、学生并发调用、typed state/response 比较和固定 gold suffix；未做 BFS/DFS、路径搜索或 LLM planning。

原始 raw token 串未保存，因而无法完全排除 parser extraction artifact；没有确认的 artifact。真实 tool/schema 错误以结构化 MCP 返回与 schema 证据为准。高不确定度和轨迹截断使本轮只达到 `EXEC-VERIFIED-PARTIAL`，不能把整个 corrected FD 集合直接当成已充分验证的 Cycle-3 训练组成。

## 结论

- Q1：YES。可执行重放显著减少 exact-sequence 的并行假阳性。
- Q2：PARTIAL。无需通用搜索即可可靠处理已证明的 same-state/gold-compatible cases；615 个仍不能判定。
- Q3：NO（在这批固定轨迹中不可观测）。re-alignment 没有后续学生轮次可继续比较。
- Q4：NOT YET。高置信子集可供下一轮设计参考，但完整 Cycle-3 训练组成仍需另行评估；本轮不设计比例、不训练。

产物位于 `run_dynamic_v1_t0/exec_verified_v2/`：`first_divergence_exec_verified_v2.jsonl`、`exec_verified_alignment_audit.parquet`、`failure_taxonomy_exec_verified_v2.json`、逐条 replay checkpoint 和完整性摘要。24 个相关 CPU 测试通过。旧 terminal-stop SHA256 为 `806597e4ba414607eaa2ebeaf0195ebee111cf0449276660f5f7ebe5fb118697`，旧 FD SHA256 为 `5304b0b4f4b3ee8527a56d7c74a0e0b338bc6db0b8a2d8f830856318bc71b7ac`，重挖后未变。

NO TRAINING · NO COMMIT · NO PUSH · NO NEW MODEL ROLLOUT
