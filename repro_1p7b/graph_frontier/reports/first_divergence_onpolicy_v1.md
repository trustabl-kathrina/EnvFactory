# First-Divergence On-Policy Correction v1 — 最终审计

结论：**局部 GO，整体 NO-GO**。严格首错 CE 使 Cycle-3 在同协议 Frozen300 上从新跑原版基线 20/300 提升至 105/300（配对净增 85，0 次成功回退），但在独立 Rich-v3 24-task heldout 上没有任何任务成功，官方 reward 在采样和贪心两种设置下均显著下降，且出现严重过量工具调用。保留 Cycle-3 checkpoint 作为 Frozen300 的研究候选；**不把它推广为通用改进，不继续训练或启动 full RL/BFCL**。原计划最多 6 轮；本次在 Cycle 3 后依据负向泛化信号提前止损。

## 1. Gold 与数据门槛

1. 原 Dynamic-v1 SFT 的 13,231 条扁平化记录不足以恢复完整 task-linked 工具依赖与真实响应；没有从文本启发式猜测 gold。旧 Guided-RL 314 条 replay-pass 中严格可用 97 条，95 条为深度 0，不适合依赖图弱桶。
2. 改用 EnvFactory Rich-v3 原始 raw + gold sidecar + replay ledger：289 FINAL_VALID，严格剔除同轮多工具、masked arguments、raw/ground-truth 不一致后，197 条可用（深度 1/2/3 为 73/77/47）。
3. 仅在学生真实 on-policy 轨迹与 gold 的工具名、结构化参数、执行成功和工具响应全部逐步一致时保留前缀；第一次错误后所有 state/action 丢弃。无可靠终止监督时不伪造 extra-tool STOP 标签。CPU 单测 18/18 通过。
4. Cycle-1 冻结 48 条，从原版 Dynamic-v1 双 GPU 采集 48/48；28 条首错可对齐并独立 EnvFactory 前缀重放 28/28。Cycle-3 又冻结与 train/heldout 不重叠的 48 条，从 Cycle-2 student 双 GPU 采集 48/48；新增 22 条首错，独立前缀重放 22/22。
5. 累计 50 个独立 task/state：premature_stop 12、wrong_arguments 20、wrong_tool 18；179,662 个 prompt/user/tool-observation token 在 loss 中遮罩，仅 2,890 个 assistant 目标 token 受训。CE 数据 SHA256 `b33c83b125e6c4b0ebfea90eb5b37f52c9dbe6dfbf12c13348b70527017e1ee9`。训练任务与 Frozen300 的 task ID 重叠 0、规范化用户 query 完全相同 0；但 37 种 CE 目标工具中 4 种也出现在 Frozen300，共 6/50 条样本，不能声称模板或工具家族完全隔离。

## 2. 训练与 checkpoint

| 阶段 | 初始化 | 训练 | 输出 SHA256 |
|---|---|---|---|
| Cycle-2 smoke | 原 Dynamic-v1 | 单 GPU、全参数 CE、16 steps、LR 1e-6；有限 loss/grad | `2f0ab72d5ec78bc5800c64d1ee86d49d8c6c3eb1e1ee06e40b7ece8de453ada4` |
| Cycle-3 bounded | Cycle-2 checkpoint | 双 GPU DDP、全参数 CE、64 global steps、LR 2e-6；有限 loss/grad；峰值 30.264 GiB | `94990703c908de8aabbe1cd8698cf0f66f6686ae95208e98d1da26dc33238100` |

Cycle-2 首次 smoke 的保存阶段因 `CUDA_HOME` 缺失失败；原失败日志保留，设置环境后同数据重跑并正确保存。没有 DPO、GRPO、PPO、外部 LLM teacher、伪造恢复轨迹或 full RL。

## 3. Frozen300：强正向，但有边界

相同冻结 manifest SHA256 `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`；300/300 均为有效 probe；同一官方 `confirm_300` evaluator、fresh MCP client/load_scenario、temperature 0、max 4 tool rounds、同样的 token/call budget。每个模型由两个独立单 GPU shard 覆盖同一批 300 任务。新跑原版 Dynamic-v1 是优先比较基准：

| 指标 | 新跑 Dynamic-v1 | Cycle-2 | Cycle-3 |
|---|---:|---:|---:|
| Task/reference-path success | 20/300 | 29/300 | **105/300** |
| Final-state success | 93/300 | 99/300 | 120/300 |
| Dependency edge complete | 94/300 | 104/300 | 200/300 |
| Gold node complete | 134/300 | 143/300 | 204/300 |

Cycle-3 对新跑基线：成功率 +28.3 个百分点；paired bootstrap 95% CI [+23.3,+33.3] pp；85 赢、0 输、215 平。Frozen300 内部 heldout 60 任务为 4→24，diagnosis 240 为 16→81。Cycle-3 对 Cycle-2 直接配对为 29→105（+76，0 输；95% CI [+20.3,+30.3] pp），因此大部分提升出现在后续 64-step 双卡训练。原历史 Dynamic-v1 成绩为 23/300；新跑同权重为 20/300，差 3 条，说明解码/运行仍有小幅非完全确定性，但远小于 +85 的净增。与历史基线比较为 23→105（+82，0 输）。

提升按环境不均匀：历史基线→Cycle-3 为 Calendar 0→23/32、GoogleTasks 12→31/80、Weather 11→49/52、TradingBot 0→2/68、UUPaoTui 0→0/68。抽查 Calendar 成功样本：原模型把上游返回的 `event_200` 错传成占位字符串；Cycle-3 正确向下游 update_event 传递 `event_200`，typed rollout 与最终状态 verifier 同时支持成功。Frozen300 是 EnvFactory 冻结任务，不是 BFCL 或跨项目外部证据；不能把这一结果外推到所有工具环境。

## 4. Rich-v3 独立 heldout：明确负向

原先预冻结 24 个 task-disjoint、Frozen300-disjoint 的 Rich-v3 任务，深度 1/2/3 各 8；两模型同 task、seed、预算且 MCP 状态隔离。

| 协议 | Dynamic-v1 semantic success | Cycle-3 semantic success | 官方 reward 均值 Dynamic-v1→Cycle-3 | 配对差值及 95% CI | 工具事件数 |
|---|---:|---:|---:|---|---:|
| 原定 temperature 0.7 | 0/24 | 0/24 | 0.18611→0.13750 | -0.04861 [-0.08333,-0.01944] | 31→101 |
| 事后温度诊断 temperature 0 | 0/24 | 0/24 | 0.18403→0.11354 | -0.07049 [-0.11632,-0.03125] | 28→115 |

第二行不是预注册的主要终点：同一任务集仅改温度、另存 plan SHA256 `0cb86095c5675146b02bf1e5951d16f7c0016b8e1007e0b13e2a42edda69e38d`，用于排除“只因 0.7 采样”解释。回退并未消失。Cycle-3 减少了一些无法解析的并行工具调用，但 wrong-tool/wrong-argument 首错与过量调用增加。缺少可靠终止 STOP target、50 条监督规模很小、Rich 任务更长且环境家族不同，均是可能解释；现有数据不能证明唯一因果。

## 5. 判定、复现与清理

- **局部 GO**：首错 on-policy CE 在 Frozen300 的参数流和任务成功指标上有可复核的大幅增益，Cycle-3 checkpoint 应保留。
- **整体 NO-GO**：Rich-v3 上 0/24 成功且两种温度下 reward 显著退步；不能启动后续 full training、RL 或宣称通用性能改善。下一阶段若要继续，先解决合法 STOP 监督和过量调用，再预冻结更大的独立 Rich heldout，而不是重复刷 Frozen300。
- 运行产物在 `repro_1p7b/logs/first_divergence_onpolicy_v1/`；两份模型在 `repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke` 和 `repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle3_ddp64`。旧结果、源数据和 frozen manifest 未覆盖。
- Scoped CPU tests：18/18；所有 `run_first_divergence_*.sh` 通过 `bash -n`。仅小型源码、测试、启动脚本及报告纳入 Git；不纳入日志、生成数据或 checkpoint。用户先前 153 个暂存文件按 path 列表 SHA256 `ac284a3a492e40241426c2e025b0f4fff8067c1c8659fc90e1c7d327fd9e0bee` 保持不变；不 push/pull/PR/reset。
- 应用层 heartbeat 间隔 90 分钟，远端本地 heartbeat 每 5 分钟记录一次；在本阶段完成后停用两者并核对 GPU 与本阶段临时进程释放。
