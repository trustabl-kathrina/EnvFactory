# First-Divergence On-Policy Correction v1 — Cycle 0

Verdict: **CONTINUE**（先取得可执行、任务级 gold，再采真实 student 前缀）。

- Dynamic SFT v1 的 13,231 条 Stage-2 Alpaca 记录已扁平化，不能直接恢复完整 task-linked gold trajectory。不能从文本猜图或猜工具响应。
- 旧 Guided-RL frozen train pool 有 314 条 replay 通过；严格排除 masked arguments 和多工具同轮后，只有 97 条可直接对齐，且 95 条深度 0，不能支撑依赖图弱桶诊断。
- 改用独立的 Rich-v3 EnvFactory 已审计池：289 条 FINAL_VALID，原始 raw、gold sidecar 和 replay ledger 俱全，Frozen300 精确 ID 重叠 0。严格单工具、无 masked-args、raw/ground-truth 一致后，可用 197 条（深度 1/2/3 为 73/77/47）。
- 已有 31 条历史 Dynamic-v1 rollout：只有 3 条满足同一严格 gold 门槛，最终得到 2 条首错纠正样本（1 个独立任务，均为 premature_stop），不足以训练。
- 对齐器只允许结构化参数 canonicalization；第一次错误后所有 state/action 丢弃；多工具同轮、无法结构化动作、gold 不可靠均跳过。核心 CPU 测试 15/15 通过。
- 独立进程每 5 分钟写 heartbeat.json/heartbeat.log；应用层对话 heartbeat 按用户更正为每 1.5 小时唤醒。

下一轮：冻结深度均衡的 48 个 Rich-v3 task，以 Dynamic-v1 做真实 on-policy rollout，先审计可用率和前缀重放；不在此轮训练。
