# First-Divergence On-Policy Correction v1 — Cycle 1

Verdict: **CONTINUE**（首错标签与独立前缀重放通过，进入 bounded CE smoke）。

## 冻结与运行

- 源：Rich-v3 已审计的 197 条严格可对齐任务；固定 48 条，深度 1/2/3 = 12/16/20，train plan SHA256 为 `2a5e43c8e417bb14d77ceab7da21931714436bca633dfcd9c993b907708fc1d1`。
- Student：原始 Dynamic-v1，双 GPU 各一个独立 SGLang server，单条真实 EnvFactory 执行轨迹。
- 首次运行的 8,192-token 服务上限拒绝了 8,457 和 11,098-token 输入（HTTP 400）；保留 10 条已落盘结果和失败日志。随后仅将推理上下文升至 32,768，冻结任务、权重和采样种子不变，续采剩余任务。
- 最终 48/48 rollout 落盘；训练前 Frozen300 exact-ID overlap = 0；gold replay certificate 48/48。

## 直接结果

| 指标 | 数值 |
|---|---:|
| Student semantic success | 1/48 |
| 官方 reward 均值 | 0.1354 |
| 首错纠正样本 / 独立任务 | 28 / 28 |
| wrong_tool / wrong_arguments / premature_stop | 14 / 8 / 6 |
| 首错 step 1 / step 2 | 21 / 7 |
| 平均首错 step（仅可用纠正） | 1.25 |
| 样本深度 1 / 2 / 3 | 9 / 10 / 9 |
| 无法结构化对齐（parallel tool calls） | 18 |
| 正确前缀观测与 gold 不一致 | 2 |
| 独立 EnvFactory 前缀重放 | 28/28 通过 |
| usable supervision coverage | 28/48 = 58.3% |
| 条件 yield（仅有效对齐 rollout） | 28/28 = 100% |

没有 perfect rollout、可靠 extra-tool STOP target 或终止动作标签；均未伪造。18 条 parallel-tool-call 和 2 条前缀不一致均未进入训练。28 条样本最多每 task 一条，错误之后的轨迹全部丢弃。

## Frontier 决策

最主要弱点是第一步错工具，其次错参数；深度 1/2/3 均有首错，不能把单一深度称为已改善。下一步仅以这 28 条通过独立重放的样本构造 assistant-only CE smoke。训练前已冻结 24 条 task-disjoint heldout（每深度 8 条），不会用 Frozen300。两条训练样本长度超过 8,192 tokens，materializer 不截断，而采用 16,384 上限；用户/工具上下文全部遮罩。
