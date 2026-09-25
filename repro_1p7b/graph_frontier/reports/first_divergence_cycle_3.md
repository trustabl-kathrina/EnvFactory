# First-Divergence On-Policy Correction v1 — Cycle 3

Verdict: **MIXED / CHECKPOINT-SELECTION**。50 条真实首错 CE 在 Frozen300 上有显著正向转移，但 Rich-v3 24-task heldout 的官方 reward 回退；因此不继续盲目训练，另用相同 Frozen300 协议核对较早的 Cycle-2 checkpoint。

## 冻结数据与训练

- Cycle-3 新冻结 48 条 Rich-v3 严格 gold 任务，与 Cycle-1 train 48 条、预冻结 heldout 24 条和 Frozen300 ID 均不重叠；深度 1/2/3 为 12/20/16。
- Cycle-2 student 双卡采集 48/48 on-policy rollout；其中 22 条首次偏离可对齐并独立 EnvFactory 前缀重放通过，错误类型为 wrong_arguments 12、premature_stop 6、wrong_tool 4。26 条含无法严格对齐的并行工具调用，均不训练。
- 与 Cycle-1 的 28 条合并后，50 条独立 task/state；premature_stop / wrong_arguments / wrong_tool = 12/20/18。179,662 个上下文 token 遮罩，2,890 个 assistant target token 参加 CE；tool-observation token 进入 loss 的数量为 0。数据 SHA256 为 `b33c83b125e6c4b0ebfea90eb5b37f52c9dbe6dfbf12c13348b70527017e1ee9`。
- 从 Cycle-2 的 Dynamic-v1 派生 smoke checkpoint 全参数续训，双 GPU DDP、64 个 global steps；loss、梯度全部有限，峰值 30.264 GiB；新权重 SHA256 为 `94990703c908de8aabbe1cd8698cf0f66f6686ae95208e98d1da26dc33238100`。没有 DPO/GRPO/PPO，也没有纠正首次偏离后的状态。

## 预冻结 Rich-v3 heldout（24 条，深度各 8）

| 指标 | Dynamic-v1 | Cycle-3 |
|---|---:|---:|
| Semantic success | 0/24 | 0/24 |
| 官方 reward 均值 | 0.18611 | 0.13750 |
| 有效结构对齐 / 首错 | 14/24 | 21/24 |
| 并行调用导致的对齐无效 | 10/24 | 3/24 |
| 工具事件数 | 31 | 101 |

配对 reward 差值 -0.04861，bootstrap 95% CI [-0.08333, -0.01944]；semantic success 差值为 0。并行调用减少是局部改善，但调用总量和 wrong-tool / wrong-argument 首错增加；在这套难度较高的 Rich 环境上没有可证明的任务成功增益。

## 独立 Frozen300（相同 manifest / 协议）

300/300 为有效 probe。与历史原版 Dynamic-v1 逐任务配对，manifest SHA256 `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`、推理参数、MCP fresh-client reset、shard 规则完全一致。两卡各执行独立 150-task shard，非共享环境。

| 指标 | Dynamic-v1 | Cycle-3 | 配对净增 |
|---|---:|---:|---:|
| Task / reference-path success | 23/300 | 105/300 | +82，95% CI [+22.3,+32.3] pp |
| Final-state success | 94/300 | 120/300 | +26，95% CI [+5.3,+12.3] pp |
| Dependency edge complete | 98/300 | 200/300 | +102，95% CI [+28.7,+39.3] pp |
| Gold-node complete | 136/300 | 204/300 | +68，95% CI [+17.3,+27.7] pp |
| Tool calls | 995 | 969 | -26 |

Task success 配对 82 赢、0 输、218 平。按 family：Calendar 0→23/32，GoogleTasks 12→31/80，Weather 11→49/52，Tradingbot 0→2/68，Uupaotui 0→0/68。随机抽查 Calendar win 的 typed rollout：原模型将返回的 `event_200` 错传成占位字符串；Cycle-3 模型将 `event_200` 正确传到 update_event，环境最终状态由 verifier 判为成功，表明至少这条提升不是汇总器假象。

## 诊断与下一步

不能把 Frozen300 的强正向结果解释成所有任务通用提升：它是 EnvFactory 生成的外部冻结集合，并非 BFCL；exact task ID 不重叠，但不能证明模板级完全独立。Rich-v3 heldout 的 reward 下降是真实冲突，可能与深度、任务复杂度、采样协议或过量调用有关，不能据此断言唯一原因。先对较早的 Cycle-2 checkpoint 运行完全相同的 Frozen300，检查 64-step 追加训练是否必要；在得到结果前不启动更多训练或新一轮纠正。
