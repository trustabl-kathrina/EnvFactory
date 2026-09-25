# First-Divergence On-Policy Correction v1 — Cycle 2

Verdict: **ADJUST-DATA**。训练链路通过，但 16-step smoke 尚无可辨认的 heldout 语义提升；下一轮增加独立 on-policy 状态并延长有限训练，不把 reward 的微小变化称为效果。

## CE 与训练

- Cycle-1 的 28/28 样本经独立 EnvFactory 前缀重放通过；28 个 task/state 均不重复；Frozen300 exact-ID overlap = 0。
- Qwen3 原生聊天模板构造 assistant tool-call target，核验 token prefix 完全一致。全体 97,925 个 prompt、user、tool-observation tokens 被标为 -100；仅 1,762 个 assistant target tokens 参与 CE loss。
- 样本最大 8,454 tokens；有 2 条超过 8,192。16,384 上限下没有截断。
- 全参数 Dynamic-v1 初始化，单卡 bf16、梯度检查点、AdamW、LR 1e-6；16 个 optimization steps，训练任务 16 个，峰值显存 25.025 GiB。16/16 loss 和梯度有限。首个尝试在保存阶段因 `CUDA_HOME` 未设置失败，日志保留；补齐环境后重跑并成功保存独立模型。loss 首/末为 0.0864/0.2431，因不同样本不可视为训练曲线改善。
- 新模型 SHA256 `2f0ab72d5ec78bc5800c64d1ee86d49d8c6c3eb1e1ee06e40b7ece8de453ada4`；与原 Dynamic-v1 权重不同，checkpoint 可由 SGLang 重载并完成真实工具调用。

## 预先冻结的 heldout 评测

24 个 Rich-v3 task，与 Cycle-1 训练 plan 完全不重叠，深度 1/2/3 各 8，Frozen300 overlap 0。两模型各在一张卡上，用相同任务、种子、温度、rollout budget 和隔离的 MCP 状态执行。事前 CPU 双环境检查确认 A 执行工具不会改变 B 的状态。

| 指标 | Dynamic-v1 | Cycle-2 smoke |
|---|---:|---:|
| Semantic success | 0/24 | 0/24 |
| Official reward 均值 | 0.17986 | 0.19028 |
| Gold-path perfect | 0/24 | 0/24 |
| 有效 first divergence | 13/24 | 13/24 |
| 无法结构化对齐的 parallel calls | 11/24 | 11/24 |
| 正确前缀观测不一致 | 0 | 0 |
| 工具执行成功 / 工具事件 | 34/38 | 34/38 |
| 对齐有效任务的平均首错 step | 1.308 | 1.385 |

配对 semantic-success 差值 0/24，95% bootstrap CI 为 [0,0]；reward 差值 +0.01042，配对 bootstrap 95% CI [-0.01250,+0.04583]，包含 0。24 个任务的动作序列均发生变化，但这不能证明能力改善；此评测太小且 success 全为 0。新模型把部分 wrong_arguments/wrong_tool 转成 premature_stop，也不能称为净提升。

## 诊断与下一步

最主要的 heldout 问题是 11/24 条 parallel tool calls 无法对齐，以及其余任务普遍在最早工具步骤偏离。16 个训练任务、16 步的 smoke 只证明实现可行，远不足以判定方法效果。下一轮从仍未使用的 Rich-v3 严格 gold pool 冻结新 task，按弱桶和深度采 Cycle-2 student 的 on-policy 轨迹，独立复验标签后与旧样本去重；若有效数据充足，再做较长但有界的双卡 CE，沿用同一 heldout 比较。不会为制造标签去破坏 student，也不会训练无法对齐的 parallel-call 状态。
