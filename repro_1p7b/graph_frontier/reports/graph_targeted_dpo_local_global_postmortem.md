# Graph-targeted DPO: local-to-global post-mortem

## Scope and provenance

Read-only analysis of one frozen 21-task paired full rollout and 29 heldout gold-prefix frontier samples, plus one new paired **next-action evaluation only** on the same 21 existing executable gold replay stop points. No training or new task/data generation. Original full-task and frontier JSONs, not aggregate logs alone, are the sources. The pilot initialization was Dynamic-v1; paired full-task IDs and recorded sampling seeds match. Checkpoints, training manifest, rollout directories, replay and gold sidecars are in the companion JSON `paths`. Raw assistant outputs, structured tool calls, typed tool events/observations, state and terminal verifier fields are in each task's `raw_rollout_file` (`steps[].model_message`, `steps[].typed_event`, `terminal_info`).

The heldout split is disjoint from preference training by task ID and has zero recorded Frozen300 exact overlap. n=21, one deterministic full rollout per model/task; uncertainty is large. The 192-task data source is a partial structural pilot, not the formal >=500-edge data gate.

## Paired official reward and success

Official mean 0.242857 → 0.187302; paired mean delta -0.055556, 95% paired bootstrap CI [-0.133333, +0.013492] (100,000 task resamples, seed 20260924). Median 0.200000 → 0.050000; ranges [0.0, 1.0] → [0.0, 0.7]. Regressions/improvements/ties: 6/2/13.

Semantic success: 1/21 → 1/21. Same successful task? **no**. Transition table (base→DPO): {'0->0': 19, '1->0': 1, '0->1': 1}. Equal totals conceal success replacement.

| task_id | base reward | DPO reward | delta | base success | DPO success |
|---|---:|---:|---:|---|---|
| gf-rich-e98b9d2c4e93985799ea | 0.4500 | 0.2500 | -0.2000 | 0 | 0 |
| gf-rich-bbf489196e02d77f46f4 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-5d8a4d55964e5379edeb | 0.4500 | 0.4500 | +0.0000 | 0 | 0 |
| gf-rich-c822bb54f76c53c0f060 | 0.2833 | 0.0833 | -0.2000 | 0 | 0 |
| gf-rich-9036b1e8a6c556cad835 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-81ecef2d56c495fa4f5e | 0.2500 | 0.5000 | +0.2500 | 0 | 0 |
| gf-rich-4a15e6d4c5090c77537d | 1.0000 | 0.5000 | -0.5000 | 1 | 0 |
| gf-rich-48a0e4c72947d4ec2a83 | 0.2500 | 0.2500 | +0.0000 | 0 | 0 |
| gf-rich-0fe310f7b1b20ddae2bc | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-3ff5ca83daf9a2a13f94 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-d638a71353678e391a99 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-7e7d9ccf76bfd6e4a037 | 0.5000 | 0.5000 | +0.0000 | 0 | 0 |
| gf-rich-d170cf277f186c2fef0b | 0.4500 | 0.0000 | -0.4500 | 0 | 0 |
| gf-rich-fd9092b992d70dab7ed7 | 0.1167 | 0.0000 | -0.1167 | 0 | 0 |
| gf-rich-43399696e495c9d900e0 | 0.2000 | 0.2000 | +0.0000 | 0 | 0 |
| gf-rich-464a4a7a73d5733061db | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-1cd0536454ecef0572a8 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-17bb56a8c624adcf70dc | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| gf-rich-da7ccea58e51eeb99ea0 | 0.5000 | 0.7000 | +0.2000 | 0 | 1 |
| gf-rich-31dedf40ba7cf16ba692 | 0.4500 | 0.4500 | +0.0000 | 0 | 0 |
| gf-rich-05941e3e3a2f64127081 | 0.2000 | 0.0500 | -0.1500 | 0 | 0 |

## Official evaluator decomposition

`guided_rl_reward.py:official_reward` computes binary subset-of-permutation exact-call trace, exact-state fraction over gold MCP servers, then `max(0, tau*trace + (1-tau)*state - length_penalty - format_penalty)`; tau=0.5. Length penalty = min(0.05 × excess tool calls over gold calls, 0.5); format penalty = min(0.05 × malformed calls, 0.5). It does **not** grade natural-language final-answer content. The graph delta is a separate shaping term and is not the official score.

| component, 21-task mean | Dynamic-v1 | DPO | delta |
|---|---:|---:|---:|
| trace_score | 0.047619 | 0.047619 | +0.000000 |
| state_score | 0.500000 | 0.500000 | +0.000000 |
| length_penalty | 0.014286 | 0.135714 | +0.121429 |
| format_penalty | 0.026190 | 0.014286 | -0.011905 |
| score | 0.242857 | 0.187302 | -0.055556 |

Reward floor (score=0): 8 → 10 tasks. Because of clamping at zero, component deltas must not simply be summed to infer each scored delta; per-task component values are in JSON/CSV.

Regression attribution: six tasks lost reward. In four, official trace and exact-state scores were unchanged while DPO made enough extra calls to increase length penalty (the base invalid-format penalty actually improved). One lost the binary official trace component by omitting the required consumer. One had both lower exact-state fraction and higher length penalty. Dataset means: trace 0.047619→0.047619; state 0.500000→0.500000; length penalty 0.014286→0.135714; format penalty 0.026190→0.014286. The global mean reward decrease is therefore driven principally by excess-call penalty plus the two task-specific trace/state regressions, not by answer-text scoring.
## Execution funnel (first required internal edge per task)

A task is counted only when the relevant observed event is present. `consumer_selected` means called after the successful producer, not necessarily successful execution. `propagation_correct` requires an inspectable returned value and matching consumer argument. Post-consumer continuation is a conditional metric among tasks with a successfully executed first consumer and a remaining gold action. The official trace is permutation-subset, while this diagnostic funnel uses the ordered gold chain.

| stage | Dynamic-v1 | DPO |
|---|---:|---:|
| producer_reached | 7/21 | 12/21 |
| producer_success | 7/21 | 10/21 |
| consumer_opportunity | 7/21 | 10/21 |
| consumer_selected | 1/21 | 6/21 |
| consumer_success | 1/21 | 6/21 |
| propagation_correct | 1/21 | 5/21 |
| downstream_continuation | 0/21 | 0/21 |
| downstream continuation conditional | 0/0 | 0/4 |
| terminal exact state (all used servers) | 7/21 | 7/21 |
| final answer emitted (not correctness) | 9/21 | 4/21 |
| official/semantic success | 1/21 | 1/21 |

Final-answer correctness cannot be extracted from the official evaluator: it scores calls and state, not response text. The table reports answer emission only; answer quality remains unknown without an independent response grader. One DPO task gets semantic/official success despite no final-answer message, showing this distinction matters.

### Earliest failure (one per task)

Base: {'initial_parser_failure:parallel_tool_calls': 11, 'consumer_not_executed': 6, 'success': 1, 'missing_producer': 3}

DPO: {'wrong_downstream_continuation': 4, 'initial_parser_failure:parallel_tool_calls': 5, 'producer_failure': 2, 'consumer_not_executed': 4, 'initial_parser_failure:generation_length_truncated': 1, 'wrong_propagation': 1, 'missing_producer': 3, 'success': 1}

These are first-edge/ordered-chain diagnostics; they do not override the official score. A first-step `parallel_tool_calls` parser rejection is **not** a premature final answer. `producer_failure` is an actual typed execution failure; `consumer_not_executed` requires a successful inspectable producer but absent consumer.

## Natural versus forced frontier

Gold-prefix forced next-action: exact 7/29 → 11/29; parameter-flow-compatible (`correct`+`ambiguous_same_tool`) 14/29 → 20/29. Paired exact wins/losses 5/1; flow wins/losses 7/1. DPO still has 18/29 exact wrong. Classes: base {'premature_stop': 11, 'ambiguous_same_tool': 7, 'correct': 7, 'wrong_value': 1, 'wrong_tool': 3}; DPO {'wrong_tool': 4, 'ambiguous_same_tool': 9, 'correct': 11, 'premature_stop': 2, 'wrong_value': 2, 'useless_retry': 1}.

Natural producer encounters are only comparable within each model's self-induced trajectory and thus have different denominators. Critically, among DPO's 8 naturally reached heldout frontier edges the reconstructed forced and natural prompt message prefixes are identical 8/8 and state snapshots equal 8/8; a context-format distribution shift at **reached** frontiers is not supported. Main differences are reach frequency and what happens after the consumer. Tool-schema order matches 7/8; natural prompt token count and gold-prefix forced token count are recorded in JSON/CSV. Base reaches 6 such edges; four prompt prefixes and five state snapshots match. This is observational and small-sample evidence, not proof that all self-induced states match.

| naturally encountered paired edge state | Base | DPO |
|---|---:|---:|
| producer_encountered_edges | 6 | 8 |
| next_action_observed_edges | 6 | 8 |
| exact | 0 | 3 |
| flow | 0 | 3 |
| exact_prefix_equal | 4 | 8 |
| state_equal | 5 | 8 |
| prior_errors | 0 | 0 |

### Frontier wins/losses with actions

Gold and model actions are preserved structurally in the companion JSON. The table uses `name:arguments` summaries and records every changed exact/flow case (so net +4/+6 must not be mistaken for only four/six gross wins).

| state | task | base class/action | DPO class/action | gold action |
|---|---|---|---|---|
| frontier-ccb39142fa7ee706ab45cc97 | gf-rich-0fe310f7b1b20ddae2bc | premature_stop / `final:{}` | ambiguous_same_tool / `CampusCard-lostCard:{"password":"123456","userId":"00123456789"}` | `CampusCard-lostCard:{"password":"student123","userId":"00123456789"}` |
| frontier-c6a6cc0f65dd4aa073e9e0af | gf-rich-da7ccea58e51eeb99ea0 | premature_stop / `final:{}` | correct / `FatSecretPlatform-get_food_details:{"food_id":"f12345"}` | `FatSecretPlatform-get_food_details:{"food_id":"f12345"}` |
| frontier-7b0f9c14157a34036db4dcac | gf-rich-48a0e4c72947d4ec2a83 | premature_stop / `final:{}` | correct / `Telecom-get_details_by_id:{"customer_id":"alice_customer_001"}` | `Telecom-get_details_by_id:{"customer_id":"alice_customer_001"}` |
| frontier-da1798208b3da8cc378e79ec | gf-rich-bbf489196e02d77f46f4 | wrong_tool / `SanvelloMentalHealthServer-start_journey:{"journey_id":"stress-management-professional","user_id":"bot_user_001"}` | correct / `Notion-notion_retrieve_user:{"format":"markdown","user_id":"bot_user_001"}` | `Notion-notion_retrieve_user:{"format":"markdown","user_id":"bot_user_001"}` |
| frontier-dca389794e72f5abf22ea5cc | gf-rich-48a0e4c72947d4ec2a83 | premature_stop / `final:{}` | ambiguous_same_tool / `MeditationServer-log_meditation_completion:{"completion_percent":100.0,"notes":"Completed meditation session for stress relief.","session_id":"session_alice_custom` | `MeditationServer-log_meditation_completion:{"completion_percent":100,"session_id":"session_alice_customer_001_001_meditate_stress_relief_1"}` |
| frontier-b6731e0126841542ce65d19d | gf-rich-9036b1e8a6c556cad835 | premature_stop / `final:{}` | correct / `Didi-taxi_cancel_order:{"order_id":"order_12345"}` | `Didi-taxi_cancel_order:{"order_id":"order_12345"}` |
| frontier-8777205b87203365f09d5405 | gf-rich-81ecef2d56c495fa4f5e | correct / `Maps-maps_place_details:{"place_id":"reverse_37.7913_-122.3915"}` | wrong_tool / `Weather-get_saved_location:{"alias":"Café Amour"}` | `Maps-maps_place_details:{"place_id":"reverse_37.7913_-122.3915"}` |
| frontier-4c38cf2f3dca5f8107c1aa4e | gf-rich-17bb56a8c624adcf70dc | premature_stop / `final:{}` | correct / `FakeStoreServer-create_product:{"category":"electronics","description":"This rare collector's edition console includes unique features such as built-in` | `FakeStoreServer-create_product:{"category":"electronics","description":"This rare collector's edition console includes unique features such as built-in` |

## Tool-call prior and shortcut

| behavior on 21 complete tasks | Base | DPO |
|---|---:|---:|
| mean_tool_calls | 1.0952380952380953 | 4.428571428571429 |
| median_tool_calls | 0 | 8 |
| max_tool_calls | 8 | 8 |
| immediate_final_tasks | 0 | 0 |
| immediate_invalid_tasks | 11 | 6 |
| immediate_parallel_invalid_tasks | 11 | 5 |
| tasks_with_final_answer | 9 | 4 |
| step_limit_tool_tasks | 1 | 11 |
| tasks_with_unexpected_tool | 4 | 8 |
| total_unexpected_tool_calls | 7 | 46 |
| total_redundant_lower_bound | 6 | 57 |
| tasks_with_adjacent_retry | 2 | 10 |
| total_adjacent_retries | 9 | 44 |
| tasks_with_failed_retry | 1 | 3 |
| total_failed_retries | 6 | 16 |
| tool_calls_after_trace_complete | 0 | 6 |

`unexpected` means not in the gold tool set; `redundant lower bound` counts tool calls beyond gold count and is not a semantic judgment. `retry` is identical adjacent call; failed retry requires the earlier execution to fail. `over-execution` is proxied by excess calls/step-limit tool calls, not a proven state-destroying extra call. There is no stored token-level logprob, so P(tool call) vs P(final) is not identifiable; observed choice frequency is the proxy.

Training preference pairs: 64 pairs / 63 states / 63 tasks / 53 consumer tools. Chosen type {'tool': 64} (tool rate 100.0%); rejected type {'tool': 12, 'final': 52} (final-answer rate 81.2%). Failure types {'wrong_value': 2, 'premature_stop': 52, 'wrong_tool': 10}; depth {'1': 50, '2': 10, '3': 4}; environment counts {'MeditationServer': 8, 'Message': 4, 'Retail': 7, 'GoogleDrive': 3, 'Airline': 6, 'HotelBooking': 1, 'FakeStoreServer': 1, 'StripePaymentServer': 2, 'Didi': 5, 'Weather': 2, 'SanvelloMentalHealthServer': 6, 'ChinaRailway': 2, 'TravelBooking': 1, 'CampusCard': 1, 'PriceComparison': 1, 'Maps': 2, 'WhatsApp': 6, 'Telecom': 7, 'TradingBot': 2, 'DrugBank': 1, 'Kuaidi100': 1, 'GoogleSheets': 5, 'PostmarkEmailService': 2, 'VehicleControl': 3, 'HowToCook': 1, 'ResendEmailService': 3, 'AirtableMcpServer': 1, 'GitHubServer': 1, 'FinancialDatasets': 1, 'Canvas': 2, 'GoogleTasks': 2, 'Notion': 3, 'Filesystem': 1, 'OneDrive': 1}. Chosen execution 100.0%. This is strong **potential action-type shortcut** evidence; it does not by itself prove a logprob shift.


## Continue-versus-stop shortcut test

Continue-required gold-prefix states (29): exact correct 7/29 → 11/29; flow-compatible 14/29 → 20/29. Gross flow wins are 4 premature-stop→exact-correct, 2 premature-stop→same-consumer/flow-correct, and 1 wrong-tool→exact-correct; one flow loss offsets these 7 wins. Thus the gain is primarily *continuation*, not isolated correction of a previously selected consumer's propagated value. Exact gross wins are five (four premature stop, one wrong tool) and one loss, net +4.

Stop-required **strict** states: 10 existing heldout gold replays both completed the gold tool chain and matched expected final environment state. On the same 10 fixed prompts and seeds, Base emitted final answer 10/10 while DPO emitted final answer 0/10 (DPO called another tool 10/10). This is direct paired evidence of an action-type shortcut at true terminal states. No response-quality grading or token logprob was used.

All 21 tool-chain-complete replay prefixes were also tested: Base action kinds {'final': 20, 'tool': 1}; DPO {'tool': 21}. Eleven replay states had `final_state_match=false` despite successful tool execution and are **excluded** from the strict stop accuracy denominator. Raw paired model messages, parsed actions, prompt SHA-256 signatures, seeds and usage are in `repro_1p7b/graph_frontier/reports/graph_targeted_dpo_stop_required_eval.json`. Both models used their existing checkpoints, temperature 0, max 512 tokens; no new task generation. The earlier natural trace-complete proxy was 1/1 versus 0/1 on different self-selected tasks and is not used as the causal stop result.

Parameter-flow-compatible DPO frontier errors remaining: nine `ambiguous_same_tool`, four wrong tool, two premature stop, two wrong value, one useless retry (18 exact errors). The full heldout frontier JSON records producer depth/consumer name, prompt/tool history, raw action and prompt-token count. Heldout gold chosen/rejected *tokenized* lengths are unavailable in original frontier state artifacts and are marked `unknown` in JSON; train-pair lengths are reported separately. No token-level P(tool) vs P(final) logprobs were stored, so only matched behavioral accuracy is claimed.

## Reward regression cases (all)

Each case below includes the query, full tool sequence, final answer, exact official reward, semantic success, and earliest identifiable failure. Additional typed per-step success/exception and numeric reward parts are in JSON. Classification is multi-label and does not assert state corruption without a demonstrated state delta.

### gf-rich-e98b9d2c4e93985799ea (Δ -0.2000)

Query: Hey, I want to create a new WhatsApp group named "New Project Team Group" for our core team members. After that, could you please remove Sarah from the "Old Project Team Group"? Finally, I need to leave the old group since we won't be using it anymore.

Gold: WhatsApp-whatsapp_create_group → WhatsApp-whatsapp_remove_group_participants → WhatsApp-whatsapp_leave_group

base: tools `INVALID:parallel_tool_calls`; reward 0.4500; success False; earliest failure initial_parser_failure:parallel_tool_calls.

Final answer: <no final answer>

dpo: tools `WhatsApp-whatsapp_create_group → WhatsApp-whatsapp_remove_group_participants → WhatsApp-whatsapp_leave_group → WhatsApp-whatsapp_get_group_info → WhatsApp-whatsapp_get_group_info → WhatsApp-whatsapp_get_group_info → WhatsApp-whatsapp_get_group_info → WhatsApp-whatsapp_get_group_info`; reward 0.2500; success False; earliest failure wrong_downstream_continuation.

Final answer: <no final answer>

Evidence labels: more_tool_calls, wrong_tool_or_unexpected_call_increase, failed_retry_loop_increase, over_execution_step_limit, correct_consumer_then_downstream_failure, length_penalty_increase

### gf-rich-c822bb54f76c53c0f060 (Δ -0.2000)

Query: Could you start by retrieving my profile details from the retail account? Once I have those, please recharge my campus card with 100 CNY using Alipay. Lastly, let's see how my meditation stats look as of now.

Gold: Retail-get_user_details → CampusCard-recharge → MeditationServer-get_meditation_stats

base: tools `INVALID:parallel_tool_calls`; reward 0.2833; success False; earliest failure initial_parser_failure:parallel_tool_calls.

Final answer: <no final answer>

dpo: tools `Retail-get_user_details → Retail-find_user_by_email → Retail-find_user_by_name_zip → Retail-list_product_types → MeditationServer-get_daily_meditation → MeditationServer-get_meditation_library → MeditationServer-get_breathing_exercises → MeditationServer-get_breathing_exercises`; reward 0.0833; success False; earliest failure producer_failure.

Final answer: <no final answer>

Evidence labels: more_tool_calls, wrong_tool_or_unexpected_call_increase, over_execution_step_limit, length_penalty_increase

### gf-rich-4a15e6d4c5090c77537d (Δ -0.5000)

Query: Can you provide me with the details of the saved location "sierra_trail", and then give me the elevation data for that specific point?

Gold: Weather-get_saved_location → Maps-maps_elevation

base: tools `Weather-get_saved_location → Maps-maps_elevation → FINAL`; reward 1.0000; success True; earliest failure success.

Final answer: The saved location "sierra_trail" has the following details:

- **Alias**: sierra_trail
- **Latitude**: 37.5987
- **Longitude**: -119.3159
- **Name**: Sierra Nevada Trailhead
- **Timezone**: America/Los_Angeles

The elevation data for this location is:

- **Elevation**: 1200.0 meters
- **Resolution**: 10.0 meters

This indicates the elevation of the Sierra Nevada Trailhead is approximately 1200 meters above sea level.

dpo: tools `Weather-get_saved_location → FINAL`; reward 0.5000; success False; earliest failure consumer_not_executed.

Final answer: <tool_call>
{"name": "Weather-maps_elevation", "arguments": {"locations": [{"latitude": 37.5987, "longitude": -119.3159}]}}
</tool_call>

Evidence labels: trace_score_regression

### gf-rich-d170cf277f186c2fef0b (Δ -0.4500)

Query: Could you please pull up the full details of my reservation for the flights from JFK to MIA on October 10th and 15th? And once I have that information, I'll need to cancel the hotel booking for The Grand Miami Hotel under booking ID booking_001.

Gold: Airline-get_reservation_details → HotelBooking-cancel_booking

base: tools `INVALID:parallel_tool_calls`; reward 0.4500; success False; earliest failure initial_parser_failure:parallel_tool_calls.

Final answer: <no final answer>

dpo: tools `Airline-get_reservation_details → HotelBooking-cancel_booking → HotelBooking-cancel_booking → HotelBooking-cancel_booking → HotelBooking-cancel_booking → HotelBooking-cancel_booking → HotelBooking-cancel_booking → HotelBooking-cancel_booking`; reward 0.0000; success False; earliest failure wrong_propagation.

Final answer: <no final answer>

Evidence labels: more_tool_calls, over_execution_step_limit, state_score_regression, length_penalty_increase

### gf-rich-fd9092b992d70dab7ed7 (Δ -0.1167)

Query: Can you fetch my current user details including all my reservation history and saved payment methods? Afterward, I'd like to change my shipping address to 123 New Oak St, Dallas, TX, USA, 75201 since I've moved into my new place. Lastly, could you provide me with a comprehensive overview of my mental health progress across all metrics?

Gold: Airline-get_user_details → Retail-modify_user_address → SanvelloMentalHealthServer-track_progress

base: tools `INVALID:parallel_tool_calls`; reward 0.1167; success False; earliest failure initial_parser_failure:parallel_tool_calls.

Final answer: <no final answer>

dpo: tools `Retail-get_user_details → Retail-find_user_by_email → Retail-find_user_by_name_zip → Retail-list_product_types → Retail-get_product_details → Retail-modify_user_address → SanvelloMentalHealthServer-track_progress → SanvelloMentalHealthServer-start_journey`; reward 0.0000; success False; earliest failure missing_producer.

Final answer: <no final answer>

Evidence labels: more_tool_calls, wrong_tool_or_unexpected_call_increase, over_execution_step_limit, length_penalty_increase

### gf-rich-05941e3e3a2f64127081 (Δ -0.1500)

Query: I need to book a return flight for myself and my colleague from SFO back to JFK on November 25th. Can you also make sure my colleague is added to the reservation as a passenger? Once that's done, could you provide me with a detailed summary of my current reservation status? Lastly, I would like to see how my recent meditation practice is progressing over the past month to ensure I'm on track.

Gold: Airline-book_reservation → Airline-update_reservation_passengers → Airline-get_reservation_details → MeditationServer-get_meditation_stats

base: tools `INVALID:parallel_tool_calls`; reward 0.2000; success False; earliest failure initial_parser_failure:parallel_tool_calls.

Final answer: <no final answer>

dpo: tools `Airline-list_airports → Airline-search_flights → Airline-book_reservation → Airline-get_reservation_details → Airline-get_user_details → MeditationServer-get_daily_meditation → MeditationServer-get_meditation_library → MeditationServer-get_meditation_library`; reward 0.0500; success False; earliest failure consumer_not_executed.

Final answer: <no final answer>

Evidence labels: more_tool_calls, wrong_tool_or_unexpected_call_increase, over_execution_step_limit, length_penalty_increase

## Training dynamics and limits

Full-parameter DPO from the original Dynamic-v1 initialization, 64 optimizer steps on two GPUs. Early/middle/late are disjoint training-log segments, not checkpoints. Only `checkpoint-64` and `final_model` exist; no intermediate step-16/32/48 frontier evaluation is possible without retraining, which was prohibited. KL/reference divergence was not logged and must be `unknown`.

| segment | loss | preference accuracy | reward margin | chosen reward | rejected reward | chosen logp | rejected logp | grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 0.3571 | 0.9286 | 1.3871 | 0.2922 | -1.0948 | -45.8333 | -79.7798 | 86.5461 |
| middle | 0.1296 | 1.0000 | 4.6536 | 0.6285 | -4.0229 | -45.4881 | -106.3333 | 19.7524 |
| late | 0.0415 | 1.0000 | 7.5312 | 0.2254 | -7.3011 | -42.3807 | -145.9773 | 7.5672 |

Final preference-val metrics: {'epoch': 2.0, 'eval_logits/chosen': -0.0602756068110466, 'eval_logits/rejected': 0.7495930790901184, 'eval_logps/chosen': -53.66666793823242, 'eval_logps/rejected': -112.91666412353516, 'eval_loss': 0.2684221863746643, 'eval_rewards/accuracies': 0.8888888955116272, 'eval_rewards/chosen': -0.3939615786075592, 'eval_rewards/margins': 5.165472984313965, 'eval_rewards/rejected': -5.562283039093018, 'eval_runtime': 16.2384, 'eval_samples_per_second': 2.155, 'eval_steps_per_second': 1.108, 'num_input_tokens_seen': 0}. Loss/margin trend alone is insufficient to establish late overfit; there is no matched mid-training heldout checkpoint.

## Direct answers and decision

Q1 — Yes, the target consumer behavior partially transfers beyond the forced benchmark: first-edge consumer selection 1/21→6/21, typed successful consumer 1/21→6/21, parameter propagation 1/21→5/21, and naturally reached heldout frontier exact 0/6→3/8 (different reach denominators). However DPO then executes the next required downstream gold action in 0/4 eligible cases, and still succeeds on only 1/21 tasks. This is Case A, not merely a forced-state artifact.

Q2 — The mean official reward point estimate truly drops 0.242857→0.187302 (paired Δ −0.055556; CI includes zero). Official trace and state means are unchanged, but excess-call length penalty rises 0.014286→0.135714 while format penalty falls 0.026190→0.014286. Four of six declining tasks keep identical trace/state; one loses trace, one loses state as well as adding excess calls. Thus over-execution/length penalty is the main recurring regression, with two discrete trajectory/state failures.

Q3 — Yes, a strong **behavioral** action-type shortcut is directly observed: 64/64 chosen actions were tools and 52/64 rejected were final answers; on 10 verifier-matched stop-required states Base stops 10/10 versus DPO 0/10. Full-task median tool calls 0→8, eight-step tool-call truncations 1→11, and final-answer emission 9→4. This establishes tool-call-over-stop behavior on these matched states, not a measured logprob shift across all contexts. Base's 11 initial `parallel_tool_calls` parser failures are not premature final answers and are explicitly separated.

Q4 — Evidence ranks: (1) objective too narrow/action-type shortcut and missing stop preservation, (2) post-consumer continuation failure and collateral wrong-tool/retry behavior, (3) upstream producer/parser errors that still limit reach. A broad forced-versus-natural context mismatch is **not** supported for the eight DPO-reached frontiers (8/8 prompt prefixes and 8/8 state snapshots match). `data too small` may contribute but cannot explain away the matched stop reversal; late overfitting is unproven without intermediate checkpoints.

PRIMARY FAILURE MECHANISM = one-step consumer preference learned continuation but generalized into tool-call-over-stop; downstream continuation remains 0/4 and extra/repeated calls incur length penalties, preventing local gains from raising end-to-end success.

NEXT ACTION = DO_NOT_SCALE_CURRENT_OBJECTIVE. Design only: mix verified continue-required, stop-required and wrong-tool preservation pairs; consider a short-horizon consumer-plus-next-gold-action preference; predeclare paired stop, downstream and official-reward gates before any new training.

NO TRAINING. NO COMMIT. NO PUSH.
