# Gold-Terminal Supervision 可行性审计

## Executive Summary

**VERDICT: PARTIAL-GO（只审计，未训练）。** 在 First-Divergence v1 严格筛出的 197 条 Rich-v3 gold 中，197 条都能识别末次工具、末次 observation 与原始的普通 assistant final 文本；没有显式 STOP/finish 动作。**但 FINAL_VALID 不等于 final state 正确**：只有 101 条的 final_state_match 为 true，其中只有 95 条同时满足 raw 工具参数、observation 与独立 replay trace 全程精确一致（深度 1/2/3 为 39/39/17）。另外 96 条 final_state_match=false，6 条虽然 verifier=true 但 raw/replay 轨迹有漂移。因此不能把 197 条直接当作可靠 terminal CE 标签，也不能宣称 95 条内容全无语义错误。

一个真实样本的 CPU Qwen3 tokenizer 验证表明：普通 assistant final 文本可作为 target；历史 user/assistant/tool tokens 全 mask，仅 target suffix 参与 CE。当前 materializer 却硬编码 gold tool_call，当前 trainer 的 manifest/schema gate 也只接受 first-divergence cumulative 数据。需要单独的 terminal materializer、schema/质量门和 CPU tests，不能直接把 JSON 并进去。

## 数据来源与锁定

- 目录：repro_1p7b/graph_frontier/preference_generation_v3/pilot_balanced_24chunks_v3。
- valid/manifest.json 锁定 289 FINAL_VALID、valid train/gold SHA、Frozen300 exact task-ID overlap=0；repro_1p7b/graph_frontier/rich_first_divergence_v1.py 的 source_bundle() 实际重验这些锁。
- valid/graph_frontier_rich_train_v1.jsonl、valid/graph_frontier_rich_gold_v1.jsonl、audit/final_ledger.jsonl、raw/*.json、replay/traces/{task_id}.rollout.json。
- 197 是 first-divergence 严格筛选：row.extra_info.graph_frontier == gold sidecar，reference_from_raw() 成功，且单工具调用/响应、typed ground_truth 对齐、无 masked_arguments、replay_ok/static_pass、无 leakage/Frozen overlap。不是全部 289。
- 另查 repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_cumulative_ce_dataset/manifest.json、dataset.jsonl，以及 cycle1/cycle3 rollouts 和 cycle2/cycle3 heldout base/trained rollout。
- 本次未产生新的 rollout、训练、checkpoint、GPU 任务；没有改动任何历史数据。

## Gold 结构、terminal state 与 target

每条 raw node.steps 为 user → [tool_call, tool_response]×N → assistant。197/197 的最后 assistant 都紧跟末次 tool_response，且 content 为非空字符串；sidecar gold_tool_sequence 与 raw tool names 一致。最后正确工具调用为第 N 个 tool_call；terminal state 是 **最后 tool_response 之后、assistant final 之前**的全部历史。未发现单独的 STOP/finish role 或特殊 token；在现有 OpenAI-compatible collector 中，无 tool_calls 且非 length 截断的普通 assistant content 被解析为 kind=final。不能凭需要而创造 STOP token。

完整模型输入的结构为：[原始 user query，先前每个 assistant tool_call 与相应 tool observation] + 同任务的 tools schema；目标为 raw 中真实存在的最后 assistant content，role=assistant、无 tool_calls。注意 raw/sidecar **没有**保存 tools JSON schema 或 call IDs；要复现模型输入必须从同任务持久 student rollout 的 tools 数组或经 CPU EnvFactory reset 获取，并使用当前 tool-chat protocol 形成 ID/role。不能宣称 raw 独立提供 byte-exact 模型 prompt。197/197 可重建结构化对话；在最严格 95 条中，53 条已有同任务持久 student rollout 的 tools schema，另外 42 条尚需 CPU schema 补齐与一致性测试。原始 user 与 train row prompt.strip() 197/197 相同。

## 197 条全量统计

| 指标 | 数量 | 判据 |
|---|---:|---|
| total_gold_usable | 197 | first-divergence 严格 reference |
| last_tool_identifiable | 197 | 最后单一 tool_call 可定位 |
| terminal_state_reconstructable | 197 结构化；53/95 可立即带持久 tools schema 实例化 | 原始消息完整，但 raw 不保存 tools/call IDs |
| gold_final_response_present | 197 | 最后 assistant 非空 |
| gold_final_response_missing | 0 | — |
| explicit_stop_present | 0 | 无显式 STOP/finish action |
| implicit_final_answer_present | 197 | 普通 assistant 文本，无 tool_calls |
| replay_ok in ledger | 197 | 仅表示可执行 replay，通过静态门 |
| final_state_match | 101 | 独立 verifier 的最终状态相符 |
| raw/replay args+observations exact | 121 | 按每步 typed JSON canonical equality |
| terminal_target_replay_valid（严格交集） | 95 | replay_ok ∧ final_state_match ∧ 全程 args/observation exact |
| terminal_target_ambiguous | 6 | verifier=true 但 raw/replay 有漂移 |
| terminal_target_invalid_for_training | 96 | final_state_match=false；不表示逐条人工判为自然语言错误 |
| 未分类 | 0 | — |

严格 95 条深度 1/2/3 为 39/39/17。全部 197 条深度 1/2/3 为 73/77/47。197 个 final 文本仅 120 种；其中 78 个完全相同，为 “The requested operations were completed successfully.”。101 条 verifier=true 中 final 文本仅 49 种。这是过度泛化或过早终止的明显风险，不能把 final_state_match 误当作 answer groundedness 验证。

## 五条真实样本

以下每例的 E「完整 input state」按实际消息顺序列出 **全部** user、assistant tool_call 和 tool observation；工具 JSON schema 在对应持久 student rollout 的 tools 字段，raw 本身不包含 schema。下述紧凑 JSON 展示与 raw 相同的字段/值；若需 byte-exact 字符串、原始换行或确定的 call IDs，使用注明的 raw/rollout 文件，不能从 sidecar 猜测。五例均在严格 95 条内。F 为原始 gold 最后一条 assistant content，未重写。

### 1. TradingBot；深度 1；2 次调用；gf-rich-0a06d52ed385e6019e82

A / E 的首条 user：I need to look up all the available technology stocks and then filter them to find which ones are priced between $50 and $150.

B / E 其余消息，按顺序：

1. assistant tool_call TradingBot-get_available_stocks，arguments={"sector":"Technology"}
2. tool observation TradingBot-get_available_stocks：{"symbols":["AAPL","GOOG","MSFT","NVDA","AMZN"]}
3. assistant tool_call TradingBot-filter_stocks_by_price，arguments={"symbols":["AAPL","GOOG","MSFT","NVDA","AMZN"],"min_price":50,"max_price":150}
4. D 最后 tool observation TradingBot-filter_stocks_by_price：{"filtered_symbols":[]}

C 为第 3 项。输入到第 4 项为止；下一目标 F 是原始 assistant 普通文本（非 STOP）：

~~~text
Based on the criteria you specified, no technology stocks are currently priced between $50 and $150. Here are the current prices of the technology stocks:

- **AAPL (Apple)**: $150.25
- **GOOG (Google)**: $2750.80
- **MSFT (Microsoft)**: $305.60
- **NVDA (Nvidia)**: $220.75
- **AMZN (Amazon)**: $3200.40

None of these fall within your specified price range.
~~~

风险：这些股价并未出现在已列出的工具观察中；即使 final_state_match=true，也不能自动视为回答事实可靠。原始 raw：pilot_balanced_24chunks_v3/raw/2026-09-24_11-10-48-31622992-sglang1.json；tools schema：repro_1p7b/logs/first_divergence_onpolicy_v1/cycle1_rollouts/gf-rich-0a06d52ed385e6019e82.r0.json 的 tools。

### 2. GoogleTasks；深度 1；2 次调用；gf-rich-7372af308535cceb8e11

A / E 的首条 user：Can you fetch the details of the task with ID '1' first, and then remove it since it's not needed right now?

B / E 其余消息：

1. assistant tool_call GoogleTasks-get_task，arguments={"task_id":"1","tasklist_id":"@default"}
2. tool observation GoogleTasks-get_task：{"task_id":"1","title":"Demo Meeting Two Weeks From Now","status":"needsAction","due":"2023-10-15T14:30:00Z","completed":null,"notes":"Confirm with team members and set up necessary tools for the demo meeting.","tasklist_id":"@default"}
3. assistant tool_call GoogleTasks-delete_task，arguments={"task_id":"1","tasklist_id":"@default"}
4. D 最后 tool observation GoogleTasks-delete_task：{}

C 为第 3 项；E 截止第 4 项。F 为原始 assistant 普通文本（非 STOP）：

~~~text
The requested operations were completed successfully.
~~~

raw：pilot_balanced_24chunks_v3/raw/2026-09-24_13-35-08-32042699-sglang.json；tools：repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_rollouts/gf-rich-7372af308535cceb8e11.r0.json。

### 3. Airline + HotelBooking；深度 2；3 次调用；gf-rich-9ec81b3b07cc5fdc5b47

A / E 的首条 user：I need to update my flight reservation to include Sarah Johnson instead of Tom Johnson. Once done, can you please proceed with canceling my hotel booking at the Grand Park Hotel as I won’t have the time to use it now?

B / E 其余消息：

1. assistant tool_call Airline-get_reservation_details，arguments={"reservation_id":"reservation_001"}
2. tool observation Airline-get_reservation_details：{"reservation_id":"reservation_001","user_id":"user_001","flights":[{"flight_number":"AA1234","date":"2023-10-06","cabin_class":"economy","total_price":280.0}],"passengers":[{"name":"Tom Johnson","seat":"12A"}],"total_price":280.0,"status":"confirmed"}
3. assistant tool_call Airline-update_reservation_passengers，arguments={"reservation_id":"reservation_001","passengers":[{"name":"Sarah Johnson","seat":"12A"}]}
4. tool observation Airline-update_reservation_passengers：{"reservation_id":"reservation_001","status":"confirmed"}
5. assistant tool_call HotelBooking-cancel_booking，arguments={"booking_id":"reservation_001"}
6. D 最后 tool observation HotelBooking-cancel_booking：{}

C 为第 5 项；E 截止第 6 项。F 为原始 assistant 普通文本（非 STOP）：

~~~text
The requested operations were completed successfully.
~~~

raw：pilot_balanced_24chunks_v3/raw/2026-09-24_12-00-09-41915995-sglang.json；tools：repro_1p7b/logs/first_divergence_onpolicy_v1/cycle1_rollouts/gf-rich-9ec81b3b07cc5fdc5b47.r0.json。

### 4. VehicleControl；深度 2；3 次调用；gf-rich-3daf1866dd5b58b6fbca

A / E 的首条 user：Could you first calculate the driving distance from Los Angeles (zipcode 90001) to Sacramento (zipcode 95814), including any intermediary cities on the route? After that, let me know if my vehicle can make this trip based on the current fuel level. If it won't be enough, fill the tank to ensure we have enough fuel for the journey.

B / E 其余消息：

1. assistant tool_call VehicleControl-estimate_distance_between_cities，arguments={"city_a_zipcode":"90001","city_b_zipcode":"95814"}
2. tool observation VehicleControl-estimate_distance_between_cities：{"distance":535.0,"intermediary_cities":[]}
3. assistant tool_call VehicleControl-estimate_drive_feasibility，arguments={"distance":535.0}
4. tool observation VehicleControl-estimate_drive_feasibility：{"can_drive":true,"current_fuel_level":35.0,"estimated_range":875.0}
5. assistant tool_call VehicleControl-fill_fuel_tank，arguments={"fuel_amount":35.0}
6. D 最后 tool observation VehicleControl-fill_fuel_tank：{"fuel_level":35.0,"error":"Fuel amount would exceed tank capacity of 50 gallons"}

C 为第 5 项；E 截止第 6 项。F 为原始 assistant 普通文本（非 STOP）：

~~~text
It appears that my previous attempt to fill the tank with 35 gallons was incorrect since it would exceed your vehicle's capacity of 50 gallons. Your vehicle currently has enough fuel for the journey from Los Angeles (zipcode 90001) to Sacramento (zipcode 95814), with a driving distance of 535 kilometers (or miles, depending on the region). You have no need to refill your tank as you already have sufficient fuel for this trip.

The estimated maximum range of your vehicle based on the current fuel level is 875 kilometers (or miles), so you should be able to reach Sacramento without any issues.
~~~

raw：pilot_balanced_24chunks_v3/raw/2026-09-23_18-34-11-40815254-sglang1.json；tools：repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_rollouts/gf-rich-3daf1866dd5b58b6fbca.r0.json。风险：末次工具返回 error；本审计的 95 条是技术 replay/verifier 子集，尚非人工确认“全部 gold 动作正确”的子集。

### 5. GoogleTasks；深度 3；4 次调用；gf-rich-16250871ff92972a670f

A / E 的首条 user：I need to double-check the task titled "Finalize Proposal Documents" that's supposed to be completed today by 5 PM. After confirming all the notes and due date are updated based on recent discussions with TechWiz Inc., I should make sure to delete those unnecessary tasks under it since they've been removed according to our meeting points. Let me start by retrieving the main task ID and task list ID first to confirm everything is accurate.

B / E 其余消息：

1. assistant tool_call GoogleTasks-get_task，arguments={"task_id":"1","tasklist_id":"1"}
2. tool observation GoogleTasks-get_task：{"task_id":"1","title":"Finalize Proposal Documents","status":"needsAction","due":"2023-11-15T17:00:00Z","completed":null,"notes":"Update the proposal documents based on recent discussions with TechWiz Inc. Remove unnecessary sub-tasks related to initial draft revisions.","tasklist_id":"1"}
3. assistant tool_call GoogleTasks-get_task_list，arguments={"tasklist_id":"1"}
4. tool observation GoogleTasks-get_task_list：{"tasklist_id":"1","title":"TechWiz Deliverables Finalization"}
5. assistant tool_call GoogleTasks-update_task，arguments={"tasklist_id":"1","task_id":"1","title":"Finalize Proposal Documents","notes":"Update the proposal documents based on recent discussions with TechWiz Inc. Remove unnecessary sub-tasks related to initial draft revisions.","due":"2023-11-15T17:00:00Z"}
6. tool observation GoogleTasks-update_task：{"task_id":"1","title":"Finalize Proposal Documents","status":"needsAction","due":"2023-11-15T17:00:00Z","completed":null,"notes":"Update the proposal documents based on recent discussions with TechWiz Inc. Remove unnecessary sub-tasks related to initial draft revisions.","tasklist_id":"1"}
7. assistant tool_call GoogleTasks-delete_task，arguments={"task_id":"2","tasklist_id":"1"}
8. D 最后 tool observation GoogleTasks-delete_task：{}

C 为第 7 项；E 截止第 8 项。F 为原始 assistant 普通文本（非 STOP）：

~~~text
The task titled "Finalize Proposal Documents" has been confirmed and updated based on recent discussions with TechWiz Inc. Unnecessary tasks under it have been removed as per the instructions. Everything is accurate and aligned with our meeting points.
~~~

raw：pilot_balanced_24chunks_v3/raw/2026-09-24_11-27-47-51868481-sglang1.json；tools：repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_rollouts/gf-rich-16250871ff92972a670f.r0.json。

## CE 语义与 First-Divergence 兼容性

现有 Type A 数据是 student on-policy 首次偏离前状态 → gold 下一 tool_call；累计 materialized 50 样本、50 tasks。现有 encode_one() 对其调用同一模型 tokenizer.apply_chat_template(..., tools=..., enable_thinking=False)，把 prompt_ids 全设 labels=-100，只对追加 assistant 目标的 suffix 计算 CE。工具历史和 observation 都在 prompt 中，故可复用这种 masking 原理。

CPU 实测一个真实 terminal 样本 gf-rich-7372af308535cceb8e11，使用其持久 tools schema、真实 gold query/调用/observation/final：prompt 2094 tokens、target suffix 9 tokens、总 2103；full_ids 的前 2094 tokens 与 add_generation_prompt 的 prompt_ids **完全相同**；suffix 为真实 final 文本加模板 <|im_end|>，不含 <tool_call>。所以 Type B 的 user、历史 assistant、tool observation 均可 mask，只训练 final assistant target；不需要新增 loss-mask 机制，也不新增 STOP token。

但是当前 materialize_first_divergence_ce.py 的 encode_one() 硬编码 target_message.tool_calls 并强制 suffix 包含 <tool_call>；train_first_divergence_ddp.py 的 load_and_schedule() 硬编码 first_divergence_cumulative_ce_v1、至少 48 行、new_examples>=20、固定 64 step 等 gate。两类样本可以共用 tokenizer、chat template、低层 input_ids/labels 及训练循环，**不能直接喂当前高层 materializer/trainer**。后续新增 target_type=gold_next_tool / gold_final_answer、独立数据 manifest 与交集质量门、重新冻结混合 schedule；不需修改已有 Type A target 的 loss mask。新增 builder 必须有 CPU tests：末次工具、末次 observation、真实 final、缺失 SKIP、observation mask、仅 target 参与 loss、无伪 STOP、replay-invalid SKIP。本轮仅审计，未新增 builder，因此没有伪造缺失样本或运行新测试。

## Gold-terminal 与 on-policy 的分布差异

Type A 的 state 是 student 真实走到首次偏离之前的 prefix；Type B 是 gold/reference 走完整路径后的 terminal state。两者不是同一 state 分布。作为第一版**独立辅助监督/ablation**在工程上可接受，但不能称纯 on-policy correction，也不能保证消除 student 的 over-calling。

现有六组 student rollout：cycle1 48、cycle3 48、cycle2 heldout base/trained 各25、cycle3 heldout base/trained 各25，共196文件，其中192条属于当前197严格参考集。按 align_first_divergence() 的逐步 gold 工具名称、参数、typed observation、state continuity 对齐规则，**perfect-prefix terminal state = 0/192**；既有 rollout 不能直接提供真正 on-policy terminal CE。应保留两类来源做消融：gold-terminal 技术合格子集与将来真实 student-perfect（目前0）；目前不能只靠后者开训。不要把 off-path 失败后的 state 当作 terminal gold。

## 数量、建议混合与风险

现有 Type A 50；Type B 原始 197（约 1:3.94），技术严格子集 95（约 1:1.90），其中持久 schema 可立即实例化 53（约 1:1.06）。但 95 仍需答案 groundedness、末次工具 error、工具路径必要性人工/CPU规则复查。候选预注册消融：Type A:Type B = 1:1、1:2、全部通过质量门的 terminal；先锁定任务去重、深度/环境/文本重复分层，再比较 over-calling 与正确工具继续率。不要把 197 无差别混入；高频通用 final 文本和部分内容缺乏 observation 支持，可能诱导过早结束。

主要限制：96 个 final-state verifier 不符；6 个 replay 漂移；42/95 无现成持久 tools schema；call ID 原始缺失导致不能逐字节重现历史 prompt；final 文本质量未独立逐条验证；零个现成 student-perfect terminal；当前 materializer/trainer gate 均为 Type A 专用。Frozen300 的 exact task-ID overlap=0 来自已锁定 manifest，此处未做新测评。FINAL_VALID 不能替代这些更严格条件。

## VERDICT

**PARTIAL-GO。** 真实 final target 与终止协议已存在，普通 CE 在 CPU tokenization 样本上可行；但可保守直接考虑的技术候选至多 95/197，且仍有 answer-quality/schema/现有 trainer 接入缺口。建议下一步先做 CPU-only terminal builder、schema 补齐和八项单测，审计 95 条语义质量及去重后冻结混合数据；完成后才讨论小规模训练，不在本轮启动。
