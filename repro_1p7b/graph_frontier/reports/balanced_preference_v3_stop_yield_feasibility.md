# Balanced Preference v3：stop 数据产率与训练门槛

结论：**DATA-NOT-READY**（不是 PIPELINE-INVALID）。本轮没有启动 DPO smoke、64-step DPO 或评测；保留原有 1219-plan 生成任务的 `PAUSED_AT_24_CHUNKS` 状态。

## 已审计数据

| 来源 | 计划数 | FINAL_VALID | 严格 stop 状态 | Dynamic-v1 真实 stop 负例 |
| --- | ---: | ---: | ---: | ---: |
| 原 24 完成分块的 v3 偏好池 | 768 | 289 | 126 | 7 |
| 独立 stop64 定向补数 | 64 | 15 | 6 | 0 |
| 独立已观察错误终止工具 stop128 定向补数 | 128 | 12 | 9 | 1 |

原偏好池在 K=4、定向 K=8 后有 513 个候选 pair：continue 319、downstream 187、stop 7。stop64 的 6 个状态 K=8 共 48 次采样全部为 final，未产生负例。stop128 的 9 个状态先 K=4、再仅对缺口 K=8，共 68 次采样：67 次 final、1 次完成后多调用工具，按每状态最多两个不同真实 rejected action 的规则仅得 1 个 stop pair。新 pair 仅用于产率诊断，尚未并入正式训练 manifest。

两个独立补数集共 27/27 条通过静态和可执行回放审计；Frozen300 精确查询重合均为 0。stop128 与历史独立评测任务的 task/query/fingerprint 排除检查为 0 条。它们各自仍标为 `PARTIAL-STRUCTURAL-DATA-READY`，不得冒充完整 Rich-v3 500-edge 发布池。

## 门槛判断

最低 280 个训练 pair 要求 stop ≥25%，即至少 70 个 stop pair；现已观察到的总数是 8 个，尚差至少 62 个，且 validation stop ≥10 个独立状态的门槛也未满足。只有通过所有数据、执行、去重、污染和序列化门槛才允许训练。

新增计划预算最多 1500 条，已使用 64+128=192 条，剩余 1308 条。新增产率为 1/192 个 stop pair/计划；按此经验产率补齐 62 个需要约 11904 条额外计划，远超过剩余上限。此为产率外推，不是数学上的不可能证明；但 stop128 已专门聚焦此前实际出现错误的四类终止工具，且仍只有 1/128 的产率。在不制造 rejected action、不提高全局 K、不改变目标函数的既定约束下，继续同类大规模生成缺乏足够可行性证据，因此停止扩张并判定 DATA-NOT-READY。

## 隔离与保留

- 正式 1219-plan 运行、旧评测、frozen300、历史模型/checkpoint 均未修改。
- stop64 和 stop128 分别具有独立 plan、resumable run、snapshot、审计与采样输出；plan/ToolGraph 哈希由 resume manifest 固定。
- stop128 计划 128/128 CPU preflight 通过，4/4 分块完成，12 组 raw/sidecar 配对落盘；生成服务及临时诊断服务已退出，两张 GPU 空闲。
- 没有 commit、push、pull、PR，也没有启动 full training。

## 下一步

若将来继续这一方向，先解决真实 stop 负例过少的问题，并在新的、事先声明的方案下重新核验 train/validation 分离及所有硬门槛；不得用合成重复 pair 或放宽 stop 比例启动当前 v3 训练。
