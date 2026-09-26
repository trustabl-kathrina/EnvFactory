# 官方 EnvFactory-RL 用作下一轮 OPD source 的审计

**VERDICT = OFFICIAL-RL-PARTIAL。** 全量静态与 executable gold replay 已完成；该数据可提供一批 FD-only 候选，但没有可验证的 gold final assistant response，平均参数槽也低于 Rich24，不能直接替代 Rich-like FD+Terminal 1:1 全流程。

## 来源和协议

- 官方数据：LARK-Lab/EnvFactory-RL，train，3092 行；主分支修订 963ec0607c016b7fafbd81570987daaf8d571153；本地文件 SHA256 0283caf6487f8790df972f192a94cdec7d17b1ccb94aceda59d21efd562105c4，与官方文件页一致。
- 原始文件仅从已有远端副本读取；无下载、模型推理、训练、Git commit 或 push。
- 真实 MCP 注册 3092/3092 行，工具存在 3085 行，参数 schema 有效 3084 行。

## Schema 与 terminal 可用性

- 3092/3092 行含 prompt/data_source/agent_name/ability/reward_model/extra_info；ground_truth 是有序 JSON 工具调用数组，含 name/arguments/masked_arguments。
- gold JSON 结构可解析 3092 行；gold 所属 server 与该行激活配置一致 3085 行；参数结构完整 3092 行；存在 2143 个 masked 参数槽。
- 独立 task_id、seed、tool observation、完整 trajectory/history、final assistant response：均未出现在发布行中。initial_config 和 final_config 均为 3092/3092。
- Gold-Terminal 分类 A explicit=0、B deterministic=0、C stable SFT join=0、D unavailable=3092。SFT-FILTERED 仅有 1 个同 query 命中，缺少配置/轨迹复合键，不能可靠恢复最终回答。

## Executable replay 与 OPD tiers

- 全量已记录 replay 3092/3092；reset 成功 2981；gold 工具链完全执行 2726；最终状态匹配 2159；严格 replay PASS 2159。
- 工具缺失/未激活服务 7 行；参数 schema 失败 1 行。7 行 gold 引用了注册表中存在但未列入自身 mcp_servers 的其他服务，属于配置缺失，不是 benign alias；不擅自补 server。
- Tier A FULL 0；Tier B FD-only 2159；Tier C structural 933；Tier D invalid 0。清洁 Tier A/B 2158，仅这些进入 source manifest。
- replay error types：{'RuntimeError': 670, 'ToolError': 255, 'static_gate_failed': 8}。工具执行失败与最终状态不匹配都不被计为 FD-ready。

## 参数、深度和环境覆盖

- 环境组合 265，MCP 服务 84；gold 调用均值/中位数 2.9056/3.0，p95=6.0，max=17。
- 每题参数槽均值/中位数 4.7743/4.0，p75=7.0，p90=10.0，max=42；含多参数调用的任务 1922/3092。
- 每个工具调用参数槽均值/中位数 1.6431/1.0，p75=2.0，p90=3.0，max=11。
- query 字符数均值/中位数 421.1067/375.0，p75=522.0，p90=697.8，max=2274。
- 调用参数数 >=2/3/4：3756/1837/823；静态 candidate-only value reuse 任务 1436，真实 observation 支持的唯一值流任务 33。两者不能混用；多数任务无完整 dependency graph，深度为 unknown。
- 清洁 Tier B 严格 parameter-rich 10；宽松 candidate-only parameter-rich 673。可验证深度分布 {'0': 535, '1': 14, '2': 1, 'unknown': 1608}。
- gold 调用长度桶 1/2/3/4+/6+ = 649/880/659/904/247；参数值类型统计 {'NoneType': 22, 'bool': 233, 'dict': 363, 'float': 359, 'int': 1942, 'list': 504, 'str': 11339}。
- 高频环境组合（前五）：[('VehicleControl', 263), ('GorillaFileSystem', 114), ('Retail', 111), ('GoogleTasks', 100), ('GitHubServer', 96)]；仅 1 题的长尾环境组合 105。各环境完整指标见 environment_statistics.json。

## 与 Rich24/Frozen300 的分布对照

| 指标 | Official RL | Rich24 | Frozen300 |
|---|---:|---:|---:|
| query 字符均值 | 421.1 | 416.0 | 123.5 |
| gold 工具调用均值 | 2.91 | 3.00 | 2.83 |
| gold 参数槽/题均值 | 4.77 | 7.08 | 6.08 |
| dependency depth | 大多数 unknown；仅报告已证实的值流深度 | 1/2/3=8/8/8 | 1/2/3=120/110/70 |
| 环境家族交集 | 84 MCP servers；与 Rich24 20/20 共用；与 Frozen300 5/5 共用 | 20 | 5 |

query 长度仅作分布代理，不能单独推断难度。官方数据覆盖广，但参数槽均值比 Rich24 低，且缺少原生 ToolGraph，不能说其依赖轨迹更深。

## 污染、去重与下一步

- Frozen300 exact query overlap 0，clean exclusion 0；Rich24 exact query overlap 0，clean exclusion 0。
- RichTrain24chunks / Generated1000 / PreferencePairs / SFT-FILTERED 的清洁排除数分别为 0/0/0/1。query hash 和环境+初始状态+完整工具参数签名分别审计，单独相同工具序列不等于同一任务。
- Generated1000 单项 gold 签名相同 53 行、单项环境+初态相同 8 行；这两种单项命中没有与 query 或彼此组成同一任务的证据，未按确切污染排除。严谨起见，下轮如用这些行训练还应做更强的近重复检测。
- 官方内部唯一 query 3092/3092；唯一完整 gold path 2963/3092；唯一工具名序列 2468/3092。
- 若下轮每个清洁 FD-ready task 只做一次 Dynamic-v1 rollout，需要约 2158 个环境 episode。现成任务可免去新一轮 14B 任务合成，但没有可靠 student runtime 基线，不提供虚构 GPU 小时。
- Q1：只能部分替代 14B generation——可替代一批 FD source，不替代 Rich-like terminal source。Q2：不能证明更丰富，平均参数少于 Rich24、依赖深度证据稀疏。Q3：PARTIAL；先用清洁 Tier B 做 on-policy FD mining，Gold-Terminal 必须另找可靠来源，本轮不启动。

## 审计局限

- No explicit final assistant response; query-only SFT overlap is not a stable join.
- Gold calls mark some argument keys as non-essential (masked_arguments); presence does not prove precision for every parameter.
- Unique typed-value flow is a conservative observation-backed proxy, not a full ToolGraph.
- Missing verified flow in multi-tool tasks means unknown depth, not depth zero.
- Strict replay requires exact selected-server final state; nonmatching official rows are not FD-ready.
- No student on-policy rollout or training was run.

NO TRAINING · NO COMMIT · NO PUSH · CPU-ONLY
