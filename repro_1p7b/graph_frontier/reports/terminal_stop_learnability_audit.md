# Terminal Stop Learnability Audit

VERDICT = TERMINAL-STOP-TRAINER-FIX

本轮只对一个真实 replay 通过的 gold terminal state 做 CPU tokenizer、preprocessing、解析器与 evaluator 源码审计。结论是：现有协议能表达并评估「立即结束、无工具调用」，但正式 terminal 编码器拒绝空 content；如果要让 loss **只**训练结束行为，还需允许 EOS-only label，并屏蔽模板尾部换行。没有训练、模型推理或新任务生成。

## 真实样本与模型

- 样本 gf-rich-7372af308535cceb8e11 来自现有 Gold-Terminal v2 数据；replay_valid=true，final_state_match=true。构造器要求每个 gold 工具执行成功、名称/参数/typed observation 与 reference 对齐；该样本末次工具 observation 为 "{}"。
- 对话角色为 user → assistant → tool → assistant → tool，最后 tool observation 之后才接 assistant target；工具 schema 共 13 个。输入 prompt 为 2074 tokens。
- 当前正式初始化模型与 tokenizer：/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b；model_type=qwen3。实际 tokenizer chat template SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8，位于该 checkpoint 的 tokenizer_config.json。
- 实际 prompt 尾部："<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"。此处 assistant 起始标记和禁用思考的 think 边界已在 prompt 内，不应计入 target。
- tokenizer EOS=<|im_end|>，id=151645；generation_config 的 EOS ids=[151645,151643]。pad 是另一 token，不是本次 label。

## 同一终态的三个真实模板渲染

| Target | assistant 内容 | target suffix | ids | target token 数 |
|---|---|---|---|---:|
| A 现有 Gold-Terminal | 原始非空 gold final | "The requested operations were completed successfully.<|im_end|>\n" | [785,11223,7525,1033,8145,7790,13,151645,198] | 9 |
| B 空 assistant | 空串 | "<|im_end|>\n" | [151645,198] | 2 |
| C 对照 | Done. | "Done.<|im_end|>\n" | [17453,13,151645,198] | 4 |

三种完整渲染都与同一个 add_generation_prompt 的 2074-token prefix 完全一致。B **确实**生成 EOS 151645，随后附带普通换行 token 198；没有生成工具调用，也不是空 target。C 只是合法性对照，不建议作为训练目标。

### B 的 token / label / loss mask（尾部）

| position | decoded token | id | 直接全 suffix labels | 建议 EOS-only labels | 建议 loss mask |
|---:|---|---:|---:|---:|---:|
| 2069 | Ċ | 198 | -100 | -100 | 0 |
| 2070 | <think> | 151667 | -100 | -100 | 0 |
| 2071 | ĊĊ | 271 | -100 | -100 | 0 |
| 2072 | </think> | 151668 | -100 | -100 | 0 |
| 2073 | ĊĊ | 271 | -100 | -100 | 0 |
| 2074 | <\|im_end\|> | 151645 | 151645 | 151645 | 1 |
| 2075 | Ċ | 198 | 198 | -100 | 0 |

attention_mask 在上述位置全部为 1。建议的唯一有效 target 是 position 2074 的 <|im_end|>，其 causal prediction 使用前一个 prompt token 的 logits；assistant-start、user 和 tool observation 都是 -100。position 2075 的换行是模板排版，不是 stop，应屏蔽。

## 正式 preprocessing 与 loss

- 正式路径：assistant_target_ce_v2.encode_terminal → encode_assistant_target → train_fd_terminal_v2_ddp.validate_mask → 训练循环中 input_ids/attention_mask/labels 传给 CausalLM CE。训练循环使用全 1 attention_mask，没有额外 collator。
- A 在正式路径通过；B 在 encode_assistant_target 被拒绝，原因为「terminal target must be existing natural language」。所以正式路径目前 **不能直接接入空 assistant**，不是 EOS 不存在。
- 用完全相同的 tokenizer.apply_chat_template 和正式 labels 规则，在独立 probe 中构造 B：prompt 全 -100、EOS 与换行均 active，validate_mask 通过，active target=2。这样会提升 EOS 概率，但同时优化 EOS 后的换行；不能称为纯 stop loss。
- 若仅 EOS active，active target=1；现有 validate_mask 的 labels[p:] == input_ids[p:] 硬性条件将其拒绝。最小修复是新增独立 terminal_stop 类型、允许空 content，并对该类型专门验证「恰好 EOS 一个有效 label；历史与尾随换行均屏蔽」；不改 tokenizer/vocab、模型或推理协议。
- CPU 解析性 CE sanity：单位置、均匀 toy logits 的 CE=11.929456，有限且非零。**这是 loss 数学路径检查，不是 Dynamic-v1 模型 forward，更不是训练或效果证明。**

## 推理停止与 AgentLoop

- 当前 Rich24/FD collector 使用 OpenAI structured message.tool_calls；tool_calls=[] 且 finish_reason 非 length 时，空 content 被分类为 kind=final。独立 probe 对空串返回 final。
- EnvFactoryBatchEnv.step 遇 kind=final 调用 _finish 并置 done=true；空 content 会在 trace 的 final_assistant_content 记为 unknown，但 _reward 只读取工具 events、ground_truth 和 final state，不读取文本。
- Frozen300 的 QueryGenNonConv.solve 在没有 tool_call 的分支写 assistant trace 并 break；parse_structured_output(空串) 得到 non_think=""、无 tool_call。QueryGen 没有要求该字符串非空。
- 实际 SGLang OpenAI request 默认 ignore_eos=false；scheduler 对 tokenizer/model EOS token 标记完成。model generation_config 包含 151645。默认 skip_special_tokens=true，因此 EOS 被剥离后返回空 content 与上述 parser 分支相容。这里是源码与配置审计，**未运行模型生成**。
- 已对安装中的 Qwen25Detector 做 CPU 空串测试：normal_text=""、calls=[]；OpenAI tool parser 不会把空回答误识别成工具调用。
- 源码定位：repro_1p7b/graph_frontier/collect_preference_rollouts.py:116-141 的 generate_action 在 calls==0 且 finish_reason!=length 时生成 final；/home/u2024311031/verl-agent/agent_system/environments/env_package/envfactory/official_envs.py:371-383 的 step 对 kind=final 调用 _finish；src/gen/query_gen/query_gen_non_conv.py:342-385 的 solve 在无 tool_call 时 break。
- 评测定位：repro_1p7b/graph_frontier/confirm_300.py:613-678 的 task_success 不检查 final_content.strip()；repro_1p7b/graph_frontier/guided_rl_reward.py:142-175 的 official_reward 只计 trace/state/penalty；安装版 SGLang 的 qwen25_detector.py:44-67、schedule_batch.py:1007-1023 分别处理空调用与 EOS 停止。

## Evaluation success 与协议 stop 分开

| 路径 | 空 assistant 是否协议性终止 | 当前 success/reward 是否要求非空 final 文本 | 证据 |
|---|---|---|---|
| Official EnvFactory（项目使用的 official reward adapter） | 是，kind=final → done | 否；官方分数是 trace/state 与格式/长度惩罚 | installed official_envs.py 的 step/_reward；guided_rl_reward.py |
| Frozen300 | 是，QueryGen 无 tool_call → break | 否；task_success 基于 valid、gold nodes、dependency edges、final state | confirm_300.py |
| Rich24 | 是，collector 空串 → final，环境 done | 否；semantic_success 基于 official trace_score 与 state_score | collect_preference_rollouts.py；audit_rich_v3_trajectories.py |

这些结论仅针对当前代码路径；尚未做真实空串模型输出的 GPU/服务端回归。尤其 trace 会把空 final_content 记为 unknown；若以后另加文本答案评分，需要重新审计，不能把「协议能 stop」等同于所有可能评价器都会接受。

## Official RL 修订解释与最小后续工作

此前 Tier A FULL=0 的定义要求 gold 自然语言 final response，仍然成立；**不覆盖旧 audit**。本轮新增的 stop-compatible 解读是：严格 replay 通过的 2,159 条可提供 gold 工具结束后的终态，清洁候选 2,158 条；理论上可构造 FD + terminal-stop，而无需 gold final 文本。因此「缺少 gold final assistant response」不再是 stop 标签的根本障碍。

但 2,159 不是即刻可训练的 terminal-stop 样本：官方发布行没有持久 typed tool observations、call IDs 或完整对话。必须从已验证 replay 重建末次 observation、工具 schema 和消息历史，再经初态/终态、工具执行、污染及长度门冻结新 source。现有 official_rl_audit/replay_tasks 只存审计摘要，不能直接当训练输入。最小变更限于独立 source builder 和 terminal_stop preprocess/mask 验证；保留原 FD 与自然语言 Gold-Terminal 数据、checkpoint、评测器不变。本轮未批量构造这 2,159 条。

Q1：YES，空 assistant 在实际模板中有 EOS target（另有换行）。Q2：NO，正式编码器拒绝空 content；独立同路径 tokenization 可监督 EOS。Q3：协议层 YES，提升 EOS 概率对应 terminal state 立即结束、无工具调用的方向，但不保证模型行为效果。Q4：当前 Official、Frozen300、Rich24 的源码成功条件均不要求非空自然语言回答；无 live 回归。Q5：FD source YES；terminal-stop source 在独立 builder + 上述小型 trainer fix 后 YES，当前不是直接可训练。

## 文件、测试与运行边界

- token-level 可复现 probe：repro_1p7b/graph_frontier/terminal_stop_audit/probe.py；结果：同目录 token_probe.json。
- 既有 CPU 测试：test_fd_terminal_v2_protocol.py 与 test_gold_terminal_builder_v2.py，20/20 PASS；独立 probe 全部断言通过。
- NO TRAINING · NO MODEL INFERENCE · NO COMMIT · NO PUSH。

