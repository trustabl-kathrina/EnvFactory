# Rich-v3 24-task Trajectory Differential Audit

## Executive conclusion

VERDICT = TOOL-POLICY-BOTTLENECK. The frozen T=0 official semantic evaluator requires both trace_score=1 and state_score=1; it does not grade final-answer wording. In this 24-task run all three primary models have trace_score=0 on every task, so no final-answer-only failure can be established as the cause of 0/24 success. Natural-language answer patterns are audited separately, without an LLM judge.

## Protocol lock and evidence

- Frozen plan SHA256: 0cb86095c5675146b02bf1e5951d16f7c0016b8e1007e0b13e2a42edda69e38d; Frozen300 manifest SHA256: 4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c.
- Verified 24/24 same task IDs and 96 rollouts with matching sampling/source seeds, initial prompt, tool schemas and initial reset state. All use the same eight-step, 1024-token, T=0 plan and collector; each collector constructs a fresh EnvFactoryBatchEnv and resets the task. The adapter generates a distinct MCP client ID from PID, reset serial, task and server. This verifies protocol equivalence, not a claim that separate processes share physical server state.
- Structured per-task diff: /home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/logs/rich_v3_trajectory_differential_audit/per_task_diff.jsonl; machine summary: /home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/logs/rich_v3_trajectory_differential_audit/summary.json.

## Model-level failure and efficiency

| Model | Tool-policy primary | Final-answer-only | Mixed/ambiguous | Mean first observed primary failure step | Tool events | Exact repeated calls | Successful/failed calls | Official required calls matched | Correct graph propagations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic-v1 | 24/24 | 0/24 | 0/24 | 1.62 | 28 | 3 | 24/4 | 8 | 3 |
| FD-only | 24/24 | 0/24 | 0/24 | 2.83 | 115 | 51 | 69/46 | 14 | 9 |
| v2-1to1 | 24/24 | 0/24 | 0/24 | 2.71 | 83 | 29 | 51/32 | 12 | 6 |
| v2-1to2 | 24/24 | 0/24 | 0/24 | 2.71 | 80 | 28 | 48/32 | 12 | 5 |

The primary-failure step is the first observed event supporting the selected mechanism, not proof that the mistake was irreversible. A separate exact gold-order divergence proxy is retained in JSON; the official trace verifier accepts a subset of a permutation of gold calls, so an early ordered divergence alone is not necessarily a failure. Provenance failure step remains unknown if the edge verifier has no step index.

- Dynamic-v1: first observed primary failure median step 1.0; distribution {1: 15, 2: 4, 3: 4, 4: 1}; gold-order proxy mean 1.17; primary mechanisms {'INVALID_TOOL_CALL': 11, 'PREMATURE_STOP': 6, 'WRONG_ARGUMENT': 7}; official trace passes 0/24, state passes 7/24.
- FD-only: first observed primary failure median step 1.5; distribution {1: 12, 2: 3, 3: 4, 6: 1, 8: 4}; gold-order proxy mean 1.21; primary mechanisms {'BUDGET_EXHAUSTED': 4, 'INVALID_TOOL_CALL': 5, 'PREMATURE_STOP': 1, 'WRONG_ARGUMENT': 14}; official trace passes 0/24, state passes 6/24.
- v2-1to1: first observed primary failure median step 1.0; distribution {1: 13, 2: 4, 3: 2, 6: 1, 8: 4}; gold-order proxy mean 1.21; primary mechanisms {'BUDGET_EXHAUSTED': 4, 'INVALID_TOOL_CALL': 7, 'PREMATURE_STOP': 1, 'WRONG_ARGUMENT': 12}; official trace passes 0/24, state passes 6/24.
- v2-1to2: first observed primary failure median step 1.0; distribution {1: 13, 2: 4, 3: 2, 6: 1, 8: 4}; gold-order proxy mean 1.21; primary mechanisms {'BUDGET_EXHAUSTED': 4, 'INVALID_TOOL_CALL': 7, 'PREMATURE_STOP': 2, 'WRONG_ARGUMENT': 11}; official trace passes 0/24, state passes 6/24.

Exact repeated calls count identical tool+argument pairs only and are a lower bound on equivalent-call loops. Official required-call matches and typed graph propagations are verifiable progress proxies. The point of the last useful environment state is not certified by current logs; extra calls after that state are marked unknown, not guessed.

## Paired Dynamic-v1 / FD-only / v2 1:1 changes

- v2 1:1 vs Dynamic-v1 official reward: better 1, same 19, worse 4; mean delta -0.02882.
- First v2-vs-base behavior difference fields: {'structured_action': 13, 'identical': 6, 'final_response': 5}. Evidence-weighted directions: {'NEUTRAL_ALTERNATIVE': 18, 'REGRESSION': 4, 'IMPROVEMENT': 1, 'AMBIGUOUS': 1}. Different valid tool paths are not marked regressions merely for differing from gold order.
- FD-only -> v2 1:1 tool events: 115 -> 83; official trace passes remain 0 -> 0.

## Terminal target contamination check

- Frozen terminal targets: 75; exact generic template count 40.
- Dynamic-v1: final actions 13/24; exact/normalized duplicate of any terminal training target 0/0; exact generic template 0; premature generic final 0.
- FD-only: final actions 7/24; exact/normalized duplicate of any terminal training target 0/0; exact generic template 0; premature generic final 0.
- v2-1to1: final actions 9/24; exact/normalized duplicate of any terminal training target 0/0; exact generic template 0; premature generic final 0.
- v2-1to2: final actions 9/24; exact/normalized duplicate of any terminal training target 0/0; exact generic template 0; premature generic final 0.

Exact and whitespace/case-normalized duplicates are textual evidence only. Non-matching wording cannot establish absence of influence, and a generic answer cannot explain an official zero score when the required tool trace is missing.

## Frozen300 versus Rich-v3 distribution

| Pool | Depth 1/2/3 | Mean gold actions | Mean distinct gold tools | Mean arguments | Mean internal edges | Multi-hop depth>=2 | Mean query chars | State mutation proxy |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Frozen300 | 120/110/70 | 2.83 | 2.83 | 6.08 | 2.37 | 180/300 | 123.5 | {'True': 247, 'False': 53} |
| Rich24 | 8/8/8 | 3.00 | 3.00 | 7.08 | 2.00 | 16/24 | 416.0 | {'True': 19, 'False': 5} |
- Frozen300 environment families: {'Calendar': 32, 'GoogleTasks': 80, 'TradingBot': 68, 'UUPaoTui': 68, 'Weather': 52}; terminal form: {'state_predicate': 127, 'state_plus_observation': 120, 'typed_observation_predicate': 53}; gold length distribution: {2: 120, 3: 110, 4: 70}.
- Rich24 environment families: {'Airline': 1, 'Airline,HotelBooking': 2, 'Airline,MeditationServer': 1, 'Airline,Retail,SanvelloMentalHealthServer': 1, 'Airline,Retail,SanvelloMentalHealthServer,Telecom': 1, 'AirtableMcpServer': 1, 'CampusCard': 1, 'CampusCard,MeditationServer': 1, 'Canvas,WhatsApp': 1, 'Didi,WhatsApp': 1, 'EbayServer,FakeStoreServer': 1, 'FakeStoreServer': 1, 'GoogleDrive': 1, 'GoogleDrive,OneDrive': 1, 'GoogleSheets': 2, 'LinkedInJobs': 1, 'MeditationServer,Notion': 1, 'Notion': 1, 'PriceComparison': 1, 'Retail': 1, 'StripePaymentServer': 1, 'WhatsApp': 1}; terminal form: {'gold_natural_language_final': 24}; gold length distribution: {2: 8, 3: 8, 4: 8}.

State mutation is only initial-versus-expected JSON inequality, not a tool-name read/write heuristic. Frozen300 uses state-predicate semantic verifiers; Rich references include natural language final text, but the **actual Rich heldout semantic flag still checks tool trace and state, not answer text**. Distribution differences alone do not prove causation.

## Per-task reward and first mechanism

| Task | Environment | Depth | Dynamic-v1 | FD-only | v2 1:1 | v2-base first behavior divergence | v2 primary failure |
|---|---|---:|---:|---:|---:|---|---|
| gf-rich-cc2b8e102fc55a4e3051 | StripePaymentServer | 1 | 0.000 | 0.000 | 0.000 | step 2 structured_action | WRONG_ARGUMENT |
| gf-rich-aea9e0cf485a58d9d05d | AirtableMcpServer | 1 | 0.450 | 0.200 | 0.450 | step None identical | INVALID_TOOL_CALL |
| gf-rich-f722cf02129be388b5f6 | FakeStoreServer | 1 | 0.000 | 0.000 | 0.000 | step 2 final_response | WRONG_ARGUMENT |
| gf-rich-8075e275da782e2a7ccb | MeditationServer,Notion | 1 | 0.250 | 0.000 | 0.250 | step 2 final_response | WRONG_ARGUMENT |
| gf-rich-d170cf277f186c2fef0b | Airline,HotelBooking | 1 | 0.450 | 0.200 | 0.200 | step 1 structured_action | WRONG_ARGUMENT |
| gf-rich-ae8044c706e625464ed8 | Airline,MeditationServer | 1 | 0.250 | 0.250 | 0.250 | step 2 final_response | PREMATURE_STOP |
| gf-rich-4430541c74efb4255730 | GoogleDrive | 1 | 0.000 | 0.000 | 0.000 | step 1 structured_action | INVALID_TOOL_CALL |
| gf-rich-464a4a7a73d5733061db | Didi,WhatsApp | 1 | 0.000 | 0.000 | 0.000 | step 2 structured_action | BUDGET_EXHAUSTED |
| gf-rich-a6fddf4d55d23818cff7 | Retail | 2 | 0.000 | 0.000 | 0.000 | step 4 final_response | WRONG_ARGUMENT |
| gf-rich-266d043368f14dc9e756 | CampusCard,MeditationServer | 2 | 0.250 | 0.000 | 0.250 | step 2 final_response | WRONG_ARGUMENT |
| gf-rich-af3eb10b7d6cc8355d01 | CampusCard | 2 | 0.450 | 0.450 | 0.450 | step None identical | INVALID_TOOL_CALL |
| gf-rich-17bb56a8c624adcf70dc | EbayServer,FakeStoreServer | 2 | 0.000 | 0.000 | 0.000 | step 4 structured_action | BUDGET_EXHAUSTED |
| gf-rich-fd9092b992d70dab7ed7 | Airline,Retail,SanvelloMentalHealthServer | 2 | 0.117 | 0.000 | 0.000 | step 1 structured_action | WRONG_ARGUMENT |
| gf-rich-4dfc5b8fb6d287d48051 | Notion | 2 | 0.000 | 0.000 | 0.000 | step 1 structured_action | WRONG_ARGUMENT |
| gf-rich-270e256b6300a3109152 | LinkedInJobs | 2 | 0.450 | 0.250 | 0.500 | step 1 structured_action | WRONG_ARGUMENT |
| gf-rich-59747dbce2e6afc81d84 | Airline,HotelBooking | 2 | 0.450 | 0.450 | 0.450 | step None identical | INVALID_TOOL_CALL |
| gf-rich-187dba701fa4cd465f53 | GoogleSheets | 3 | 0.000 | 0.000 | 0.000 | step None identical | INVALID_TOOL_CALL |
| gf-rich-8099908b1db355ac626d | Airline | 3 | 0.000 | 0.000 | 0.000 | step 3 structured_action | WRONG_ARGUMENT |
| gf-rich-12c89076a8117ef702d4 | WhatsApp | 3 | 0.000 | 0.000 | 0.000 | step None identical | INVALID_TOOL_CALL |
| gf-rich-9b0042bd254db4f4be41 | GoogleSheets | 3 | 0.000 | 0.000 | 0.000 | step 3 structured_action | BUDGET_EXHAUSTED |
| gf-rich-e315d44625e9c6e615a1 | PriceComparison | 3 | 0.500 | 0.500 | 0.500 | step 2 structured_action | WRONG_ARGUMENT |
| gf-rich-74e19697b936833a863c | Airline,Retail,SanvelloMentalHealthServer,Telecom | 3 | 0.450 | 0.175 | 0.175 | step 1 structured_action | BUDGET_EXHAUSTED |
| gf-rich-6e94d333418bada4070e | GoogleDrive,OneDrive | 3 | 0.150 | 0.050 | 0.050 | step 7 structured_action | WRONG_ARGUMENT |
| gf-rich-3639d42de3698e1df765 | Canvas,WhatsApp | 3 | 0.200 | 0.200 | 0.200 | step None identical | INVALID_TOOL_CALL |

## Five largest gains and five largest regressions

### Largest gains

#### gf-rich-270e256b6300a3109152 (delta +0.050; IMPROVEMENT)

Query: I came across an interesting job listing for a Senior Software Developer position at TechWave Innovations. Could you provide me with more details on this specific job, including its full description, requirements, and application process? Additionally, can you share some background information on TechWave Innovations, such as their mission statement, industry focus, size, headquarters location, and official website? Lastly, I would appreciate if you could look up any other current openings they have available at the moment.

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 2:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 3:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 4:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 5:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 6:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 7:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 8:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown]
- v2 1:1: 1:LinkedInJobs-get_job_details({"job_id": "1234567890"}) [exec=False, obs=unknown] -> 2:FINAL(The job details could not be retrieved. Please provide the correct job ID for the specific listing.)
- First structured difference: step 1, structured_action; base={'kind': 'invalid', 'raw': '{"content":null,"reasoning_content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"keywords\\": \\"Senior Software Developer\\", \\"location\\": \\"United States\\", \\"job_type\\": \\"FULL_T...; v2={'arguments': {'job_id': '1234567890'}, 'kind': 'tool', 'name': 'LinkedInJobs-get_job_details'}.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 1.0/1.0.

#### gf-rich-f722cf02129be388b5f6 (delta +0.000; NEUTRAL_ALTERNATIVE)

Query: I want to get the detailed information for the SmartWear HeartGuard Watch by its ID first, so I can then add it to our virtual store. Could you fetch the details for me, please? And once I have that, let's go ahead and add the product to our store with all the specifics included.

- Dynamic-v1: 1:FakeStoreServer-get_product_by_id({"product_id": 12345}) [exec=False, obs=unknown] -> 2:FINAL(It seems there might be an issue with the product ID provided. Let me check the available products to see if the SmartWear HeartGuard Watch exists. Could you co...)
- FD-only: 1:FakeStoreServer-get_product_by_id({"product_id": 12345}) [exec=False, obs=unknown] -> 2:FINAL(The product with ID 12345 was not found. Let me check if there's another way to retrieve the details for the SmartWear HeartGuard Watch. Could you please confir...)
- v2 1:1: 1:FakeStoreServer-get_product_by_id({"product_id": 12345}) [exec=False, obs=unknown] -> 2:FINAL(The product with ID 12345 was not found. Let me check if there's another way to retrieve the details for the SmartWear HeartGuard Watch. Could you please provid...)
- First structured difference: step 2, final_response; base=It seems there might be an issue with the product ID provided. Let me check the available products to see if the SmartWear HeartGuard Watch exists. Could you confirm the correct product ID or provide more details about the product?; v2=The product with ID 12345 was not found. Let me check if there's another way to retrieve the details for the SmartWear HeartGuard Watch. Could you please provide the product ID or the name of the product?.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 0.0/0.0.

#### gf-rich-e315d44625e9c6e615a1 (delta +0.000; AMBIGUOUS)

Query: Can you find the product identifier and position for the Gaming Laptop Model XYZ so I can look up any discounts at Best Buy and see how the prices compare across different retailers? Also, let's check the predicted price trend for this laptop as well.

- Dynamic-v1: 1:PriceComparison-get_pid_pos_details({"name": "Gaming Laptop Model XYZ"}) [exec=True, obs={'name': 'Gaming Laptop Model XYZ', 'pid': 'LPT1234567890', 'pos': 1}] -> 2:FINAL(The product identifier for the Gaming Laptop Model XYZ is **LPT1234567890**, and its position is **1**.   Now, I will retrieve the discount and promotional info...)
- FD-only: 1:PriceComparison-get_pid_pos_details({"name": "Gaming Laptop Model XYZ"}) [exec=True, obs={'name': 'Gaming Laptop Model XYZ', 'pid': 'LPT1234567890', 'pos': 1}] -> 2:PriceComparison-compare_price({"pid": "LPT1234567890", "pos": 1}) [exec=True, obs={'pid': 'LPT1234567890', 'pos': 1, 'retailers': [{'availability': 'In Stock', 'price': 159...] -> 3:PriceComparison-get_discount_info({"pid": "LPT1234567890", "pos": 1, "retailer": "Best Buy", "time_period": "current"}) [exec=True, obs={'discounts': [{'discount_percentage': 15.0, 'discounted_price': 1545.42, 'end_date': '202...] -> 4:PriceComparison-price_trend({"pid": "LPT1234567890", "pos": 1}) [exec=True, obs={'pid': 'LPT1234567890', 'pos': 1, 'price_data': [{'date': '2023-04-15', 'predicted_price'...] -> 5:FINAL(The product identifier for the Gaming Laptop Model XYZ is **LPT1234567890**, and its position is **1**.   Here are the details for Best Buy: - **Discount**: 15%...)
- v2 1:1: 1:PriceComparison-get_pid_pos_details({"name": "Gaming Laptop Model XYZ"}) [exec=True, obs={'name': 'Gaming Laptop Model XYZ', 'pid': 'LPT1234567890', 'pos': 1}] -> 2:PriceComparison-compare_price({"pid": "LPT1234567890", "pos": 1}) [exec=True, obs={'pid': 'LPT1234567890', 'pos': 1, 'retailers': [{'availability': 'In Stock', 'price': 159...] -> 3:PriceComparison-get_discount_info({"pid": "LPT1234567890", "pos": 1, "retailer": "Best Buy", "time_period": "current"}) [exec=True, obs={'discounts': [{'discount_percentage': 15.0, 'discounted_price': 1545.42, 'end_date': '202...] -> 4:PriceComparison-price_trend({"pid": "LPT1234567890", "pos": 1}) [exec=True, obs={'pid': 'LPT1234567890', 'pos': 1, 'price_data': [{'date': '2023-04-15', 'predicted_price'...] -> 5:FINAL(The product identifier for the Gaming Laptop Model XYZ is **LPT1234567890** and its position is **1**.   Here are the current prices across different retailers:...)
- First structured difference: step 2, structured_action; base={'content': "The product identifier for the Gaming Laptop Model XYZ is **LPT1234567890**, and its position is **1**. \n\nNow, I will retrieve the discount and promotional information for this product at Best Buy and compare the prices acros...; v2={'arguments': {'pid': 'LPT1234567890', 'pos': 1}, 'kind': 'tool', 'name': 'PriceComparison-compare_price'}.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 1/3; official state score base/v2 1.0/1.0.

#### gf-rich-cc2b8e102fc55a4e3051 (delta +0.000; NEUTRAL_ALTERNATIVE)

Query: Can you check the status of the previous payment intent ID "pi_123456" from Sarah's account and then create a new payment intent for her new order using the same payment details?

- Dynamic-v1: 1:StripePaymentServer-retrieve_payment_intent({"payment_intent_id": "pi_123456"}) [exec=True, obs={'amount': 9900, 'client_secret': '[redacted]', 'created': 1628768400, 'currency': 'usd', ...] -> 2:FINAL(The previous payment intent "pi_123456" has been successfully processed. The status is **"succeeded"**.  Now, I will create a new payment intent for Sarah's new...)
- FD-only: 1:StripePaymentServer-retrieve_payment_intent({"payment_intent_id": "pi_123456"}) [exec=True, obs={'amount': 9900, 'client_secret': '[redacted]', 'created': 1628768400, 'currency': 'usd', ...] -> 2:StripePaymentServer-create_payment_intent({"amount": 9900, "currency": "usd", "metadata": {"order_details": "Eco-Friendly Cleaning Bundle"}, "payment_me...) [exec=True, obs={'amount': 9900, 'client_secret': '[redacted]', 'created': 1628749800, 'currency': 'usd', ...] -> 3:FINAL(The previous payment intent "pi_123456" has been successfully retrieved and confirmed as completed. A new payment intent "pi_2" has been created using the same ...)
- v2 1:1: 1:StripePaymentServer-retrieve_payment_intent({"payment_intent_id": "pi_123456"}) [exec=True, obs={'amount': 9900, 'client_secret': '[redacted]', 'created': 1628768400, 'currency': 'usd', ...] -> 2:StripePaymentServer-create_payment_intent({"amount": 9900, "currency": "usd", "customer_id": "cus_123456", "metadata": {"order_details": "Eco-Friendly C...) [exec=True, obs={'amount': 9900, 'client_secret': '[redacted]', 'created': 1628749800, 'currency': 'usd', ...] -> 3:FINAL(The new payment intent "pi_2" has been created successfully. It is currently in a state of "requires_payment_method", which means the customer needs to provide ...)
- First structured difference: step 2, structured_action; base={'content': 'The previous payment intent "pi_123456" has been successfully processed. The status is **"succeeded"**.\n\nNow, I will create a new payment intent for Sarah\'s new order using the same payment details. Let\'s proceed with that....; v2={'arguments': {'amount': 9900, 'currency': 'usd', 'customer_id': 'cus_123456', 'metadata': {'order_details': 'Eco-Friendly Cleaning Bundle'}, 'payment_method_id': 'pm_123456'}, 'kind': 'tool', 'name': 'StripePaymentServer-create_payment_int....
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 1/1; official state score base/v2 0.0/0.0.

#### gf-rich-af3eb10b7d6cc8355d01 (delta +0.000; NEUTRAL_ALTERNATIVE)

Query: Sure, could you first check my current balance? Then, let's make sure all my personal information is up-to-date — like my phone number and email. Lastly, I'd like to see the transaction history from the last month so I can keep track of everything properly.

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:INVALID(parallel_tool_calls)
- v2 1:1: 1:INVALID(parallel_tool_calls)
- First structured difference: step None, identical; base=None; v2=None.
- Primary v2 failure: INVALID_TOOL_CALL; invalid terminal action while required calls missing.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 1.0/1.0.

### Largest regressions

#### gf-rich-74e19697b936833a863c (delta -0.275; REGRESSION)

Query: Since I need to book a last-minute flight for a work meeting, can you first find my user details at the retail store using my email alice.smith@example.com? Then, using the customer ID from that, please get all the detailed information available on my telecom account including billing statements and other contacts. After that, could you check my user profile on the airline's service using my unique user ID to see if I have any saved payment methods? Lastly, please give me an update on my progress in managing my mental health with Sanvello, as I want to make sure I'm staying mindful of my well-being while preparing for this trip.

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:Retail-find_user_by_email({"email": "alice.smith@example.com"}) [exec=True, obs={'email': 'alice.smith@example.com', 'name': 'Alice Smith', 'user_id': 'alice987', 'zip_co...] -> 2:Retail-get_user_details({"user_id": "alice987"}) [exec=True, obs={'address': {'city': 'Boston', 'state': 'MA', 'street1': '23 Elm Street', 'zip_code': '021...] -> 3:Telecom-get_customer_by_phone({"phone_number": "123-456-7890"}) [exec=False, obs=unknown] -> 4:Telecom-get_customer_by_id({"customer_id": "alice987"}) [exec=True, obs={'customer_id': 'alice987', 'full_name': 'Alice Smith', 'line_ids': ['line_001'], 'phone_n...] -> 5:SanvelloMentalHealthServer-track_progress({"user_id": "alice987"}) [exec=True, obs={'journey_progress': [{'current_step': 3, 'enrollment_id': 'enrollment_1', 'journey_id': '...] -> 6:Airline-book_reservation({"cabin": "business", "destination": "LAX", "flight_type": "business", "flights": [{"arrival_time": "2023-10-1...) [exec=True, obs={'reservation_id': 'RES3D2AE5A9', 'status': 'confirmed', 'total_price': 15.0}] -> 7:Airline-get_flight_status({"date": "2023-10-15", "flight_number": "AA123"}) [exec=True, obs={'date': '2023-10-15', 'flight_number': 'AA123', 'status': 'unknown'}] -> 8:Airline-cancel_reservation({"reservation_id": "RES3D2AE5A9"}) [exec=True, obs={'refund_amount': 12.0, 'reservation_id': 'RES3D2AE5A9', 'status': 'cancelled'}]
- v2 1:1: 1:Retail-find_user_by_email({"email": "alice.smith@example.com"}) [exec=True, obs={'email': 'alice.smith@example.com', 'name': 'Alice Smith', 'user_id': 'alice987', 'zip_co...] -> 2:Retail-get_user_details({"user_id": "alice987"}) [exec=True, obs={'address': {'city': 'Boston', 'state': 'MA', 'street1': '23 Elm Street', 'zip_code': '021...] -> 3:Telecom-get_customer_by_phone({"phone_number": "123-456-7890"}) [exec=False, obs=unknown] -> 4:Telecom-get_customer_by_id({"customer_id": "alice987"}) [exec=True, obs={'customer_id': 'alice987', 'full_name': 'Alice Smith', 'line_ids': ['line_001'], 'phone_n...] -> 5:SanvelloMentalHealthServer-track_progress({"user_id": "alice987"}) [exec=True, obs={'journey_progress': [{'current_step': 3, 'enrollment_id': 'enrollment_1', 'journey_id': '...] -> 6:Airline-book_reservation({"cabin": "business", "destination": "LAX", "flight_type": "business", "flights": [{"arrival_time": "2023-10-1...) [exec=True, obs={'reservation_id': 'RES26045BAA', 'status': 'confirmed', 'total_price': 15.0}] -> 7:Airline-get_flight_status({"date": "2023-10-15", "flight_number": "AA123"}) [exec=True, obs={'date': '2023-10-15', 'flight_number': 'AA123', 'status': 'unknown'}] -> 8:Airline-cancel_reservation({"reservation_id": "RES26045BAA"}) [exec=True, obs={'refund_amount': 12.0, 'reservation_id': 'RES26045BAA', 'status': 'cancelled'}]
- First structured difference: step 1, structured_action; base={'kind': 'invalid', 'raw': '{"content":null,"reasoning_content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"email\\": \\"alice.smith@example.com\\"}","name":"Retail-find_user_by_email"},"id":"call_76b8e3d0262b419db01...; v2={'arguments': {'email': 'alice.smith@example.com'}, 'kind': 'tool', 'name': 'Retail-find_user_by_email'}.
- Primary v2 failure: BUDGET_EXHAUSTED; eight-step budget reached with required calls missing.
- Mechanism evidence: required calls matched base/v2 0/2; official state score base/v2 1.0/0.75.

#### gf-rich-d170cf277f186c2fef0b (delta -0.250; REGRESSION)

Query: Could you please pull up the full details of my reservation for the flights from JFK to MIA on October 10th and 15th? And once I have that information, I'll need to cancel the hotel booking for The Grand Miami Hotel under booking ID booking_001.

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:Airline-get_reservation_details({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 2:Airline-get_user_details({"user_id": "user_001"}) [exec=True, obs={'name': 'Maria Sanchez', 'payment_methods': ['pm_001'], 'reservations': ['reservation_001...] -> 3:Airline-get_reservation_details({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 4:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 5:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 6:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 7:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 8:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown]
- v2 1:1: 1:Airline-get_reservation_details({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 2:Airline-get_user_details({"user_id": "user_001"}) [exec=True, obs={'name': 'Maria Sanchez', 'payment_methods': ['pm_001'], 'reservations': ['reservation_001...] -> 3:Airline-get_reservation_details({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 4:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 5:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 6:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 7:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown] -> 8:Airline-cancel_reservation({"reservation_id": "booking_001"}) [exec=False, obs=unknown]
- First structured difference: step 1, structured_action; base={'kind': 'invalid', 'raw': '{"content":null,"reasoning_content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"reservation_id\\": \\"booking_001\\"}","name":"Airline-get_reservation_details"},"id":"call_73ff3f5e959540aa...; v2={'arguments': {'reservation_id': 'booking_001'}, 'kind': 'tool', 'name': 'Airline-get_reservation_details'}.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 1.0/1.0.

#### gf-rich-fd9092b992d70dab7ed7 (delta -0.117; REGRESSION)

Query: Can you fetch my current user details including all my reservation history and saved payment methods? Afterward, I'd like to change my shipping address to 123 New Oak St, Dallas, TX, USA, 75201 since I've moved into my new place. Lastly, could you provide me with a comprehensive overview of my mental health progress across all metrics?

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:Retail-get_user_details({"user_id": "1234567890"}) [exec=False, obs=unknown] -> 2:Retail-find_user_by_email({"email": "user@example.com"}) [exec=False, obs=unknown] -> 3:Retail-find_user_by_name_zip({"first_name": "John", "last_name": "Doe", "zip_code": "75201"}) [exec=False, obs=unknown] -> 4:Retail-list_product_types({}) [exec=True, obs={'products': [{'name': 'Travel Backpack', 'product_id': 'prod-1'}]}] -> 5:Retail-get_product_details({"product_id": "prod-1"}) [exec=True, obs={'description': 'A durable backpack for travelers, suitable for carrying daily essentials ...] -> 6:Retail-modify_user_address({"address": "123 New Oak St", "city": "Dallas", "country": "USA", "state": "TX", "user_id": "1234567890", "zip...) [exec=False, obs=unknown] -> 7:SanvelloMentalHealthServer-track_progress({"user_id": "1234567890"}) [exec=True, obs={'journey_progress': [], 'mood_trends': {'best_day': 'Monday', 'common_triggers': ['work s...] -> 8:SanvelloMentalHealthServer-start_journey({"journey_id": "journey-1", "user_id": "1234567890"}) [exec=False, obs=unknown]
- v2 1:1: 1:Retail-get_user_details({"user_id": "1234567890"}) [exec=False, obs=unknown] -> 2:Retail-find_user_by_email({"email": "user@example.com"}) [exec=False, obs=unknown] -> 3:Retail-find_user_by_name_zip({"first_name": "John", "last_name": "Doe", "zip_code": "75201"}) [exec=False, obs=unknown] -> 4:Retail-list_product_types({}) [exec=True, obs={'products': [{'name': 'Travel Backpack', 'product_id': 'prod-1'}]}] -> 5:Retail-get_product_details({"product_id": "prod-1"}) [exec=True, obs={'description': 'A durable backpack for travelers, suitable for carrying daily essentials ...] -> 6:Retail-modify_user_address({"address": "123 New Oak St", "city": "Dallas", "country": "USA", "state": "TX", "user_id": "1234567890", "zip...) [exec=False, obs=unknown] -> 7:SanvelloMentalHealthServer-track_progress({"user_id": "1234567890"}) [exec=True, obs={'journey_progress': [], 'mood_trends': {'best_day': 'Monday', 'common_triggers': ['work s...] -> 8:Retail-get_user_details({"user_id": "1234567890"}) [exec=False, obs=unknown]
- First structured difference: step 1, structured_action; base={'kind': 'invalid', 'raw': '{"content":null,"reasoning_content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"user_id\\": \\"1234567890\\"}","name":"Retail-get_user_details"},"id":"call_7b3c508162594396beec277d","index...; v2={'arguments': {'user_id': '1234567890'}, 'kind': 'tool', 'name': 'Retail-get_user_details'}.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 0.3333333333333333/0.3333333333333333.

#### gf-rich-6e94d333418bada4070e (delta -0.100; REGRESSION)

Query: I need to make sure I have the most recent version of the animated mock-up by downloading it from Google Drive. After reviewing and updating it locally, I want to upload the latest version to OneDrive so I can double-check it there before making any additional changes. Once that’s done, I’ll also want to download the updated file from OneDrive for one last offline review. Finally, since I know I'll be dealing with a big video file soon, I should upload that using OneDrive's dedicated resumable upload feature.

- Dynamic-v1: 1:GoogleDrive-list_files({"q": "name=\"animated mock-up\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 2:GoogleDrive-download_file({"acknowledge_abuse": false, "file_id": "animated_mockup_12345_gdrive"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'mimeType': 'image/gif', 'name': '...] -> 3:OneDrive-upload_large_file({"content": "UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...", "name": "Animated_Mockup.gif"}) [exec=True, obs={'id': 'file_3', 'name': 'Animated_Mockup.gif', 'size': 40, 'upload_session_id': 'session_...] -> 4:OneDrive-download_file({"item_id": "file_3"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'content_type': 'application/octet...] -> 5:GoogleDrive-list_files({"q": "name=\"Animated_Mockup.gif\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 6:OneDrive-list_children({"item_id": "file_3", "top": 1}) [exec=True, obs={'value': []}] -> 7:FINAL(<tool_call> {"name": "OneDrive-list_files", "arguments": {"q": "name=\"Animated_Mockup.gif\" AND mimeType=\"video/mp4\""}} </tool_call>)
- FD-only: 1:GoogleDrive-list_files({"q": "name=\"animated mock-up\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 2:GoogleDrive-download_file({"acknowledge_abuse": false, "file_id": "animated_mockup_12345_gdrive"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'mimeType': 'image/gif', 'name': '...] -> 3:OneDrive-upload_large_file({"content": "UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...", "name": "Animated_Mockup.gif"}) [exec=True, obs={'id': 'file_3', 'name': 'Animated_Mockup.gif', 'size': 40, 'upload_session_id': 'session_...] -> 4:OneDrive-download_file({"item_id": "file_3"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'content_type': 'application/octet...] -> 5:GoogleDrive-list_files({"q": "name=\"Animated_Mockup.gif\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 6:OneDrive-list_children({"item_id": "file_3", "top": 1}) [exec=True, obs={'value': []}] -> 7:OneDrive-upload_large_file({"content": "UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...", "name": "Animated_Mockup.gif"}) [exec=True, obs={'id': 'file_4', 'name': 'Animated_Mockup.gif', 'size': 40, 'upload_session_id': 'session_...] -> 8:OneDrive-download_file({"item_id": "file_4"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'content_type': 'application/octet...]
- v2 1:1: 1:GoogleDrive-list_files({"q": "name=\"animated mock-up\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 2:GoogleDrive-download_file({"acknowledge_abuse": false, "file_id": "animated_mockup_12345_gdrive"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'mimeType': 'image/gif', 'name': '...] -> 3:OneDrive-upload_large_file({"content": "UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...", "name": "Animated_Mockup.gif"}) [exec=True, obs={'id': 'file_3', 'name': 'Animated_Mockup.gif', 'size': 40, 'upload_session_id': 'session_...] -> 4:OneDrive-download_file({"item_id": "file_3"}) [exec=True, obs={'content': 'UEsDBBQABgAIAAAAIQDZjRl67wEAALQAFAAAS...', 'content_type': 'application/octet...] -> 5:GoogleDrive-list_files({"q": "name=\"Animated_Mockup.gif\" AND mimeType=\"video/mp4\""}) [exec=True, obs={'files': [{'id': 'animated_mockup_12345_gdrive', 'mimeType': 'image/gif', 'modifiedTime':...] -> 6:OneDrive-list_children({"item_id": "file_3", "top": 1}) [exec=True, obs={'value': []}] -> 7:OneDrive-list_children({"item_id": "file_3", "top": 1}) [exec=True, obs={'value': []}] -> 8:OneDrive-list_children({"item_id": "file_3", "top": 1}) [exec=True, obs={'value': []}]
- First structured difference: step 7, structured_action; base={'content': '<tool_call>\n{"name": "OneDrive-list_files", "arguments": {"q": "name=\\"Animated_Mockup.gif\\" AND mimeType=\\"video/mp4\\""}}\n</tool_call>', 'kind': 'final', 'raw': '<tool_call>\n{"name": "OneDrive-list_files", "arguments": ...; v2={'arguments': {'item_id': 'file_3', 'top': 1}, 'kind': 'tool', 'name': 'OneDrive-list_children'}.
- Primary v2 failure: WRONG_ARGUMENT; required tool attempted without matching required arguments.
- Mechanism evidence: required calls matched base/v2 2/2; official state score base/v2 0.5/0.5.

#### gf-rich-12c89076a8117ef702d4 (delta +0.000; NEUTRAL_ALTERNATIVE)

Query: Okay, I'll mark the latest message from Mark as read so I don't forget about it. Then, I'll archive the Project Team chat for now to focus on other tasks without getting distracted. Afterward, I should update Lisa with my new mobile number. Finally, I'll send a quick reminder to the team about reviewing progress reports before Friday.

- Dynamic-v1: 1:INVALID(parallel_tool_calls)
- FD-only: 1:INVALID(parallel_tool_calls)
- v2 1:1: 1:INVALID(parallel_tool_calls)
- First structured difference: step None, identical; base=None; v2=None.
- Primary v2 failure: INVALID_TOOL_CALL; invalid terminal action while required calls missing.
- Mechanism evidence: required calls matched base/v2 0/0; official state score base/v2 0.0/0.0.

## Q1–Q8 direct answers

- Q1: v2 1:1 most frequent primary mechanism is WRONG_ARGUMENT (12/24); all 24 have official trace failure.
- Q2: terminal supervision reduces FD-only tool events 115 -> 83 (27.8%); 6/24 tasks use fewer calls. Exact repeats fall 51 -> 29, but trace passes remain zero.
- Q3: the observed bottleneck for this official benchmark is tool policy, 24/24 per model; final-answer grounding remains ungraded by the semantic flag and cannot be exonerated or blamed here.
- Q4: first observed primary-failure mean steps base/FD/v2 1.62/2.83/2.71. This mechanism-conditioned statistic is not an identical-cause survival analysis; the stricter gold-order proxy means are listed above.
- Q5: local transfer is limited but real as a proxy: official required-call matches 8 -> 12; correct graph propagations 3 -> 6. Paired required-call deltas {0: 22, 2: 2}; paired propagation deltas {1: 1, 0: 22, 2: 1}. Neither is a task-level success gain.
- Q6: 40/75 terminal targets share one exact generic response, but v2 has 0 exact and 0 normalized target duplicates; repeated normalized complete final texts 0; premature exact-generic finals 0. There is no direct textual-copy evidence; subtler influence remains unknown.
- Q7: Rich24 spans mostly different environment families, has balanced depth 8/8/8 versus Frozen300 120/110/70, longer queries (416.0 vs 123.5 chars), and more gold arguments (7.08 vs 6.08); a distribution gap is measured, not proven sufficient to cause failure.
- Q8: among the proposed choices, prioritize C, deeper/longer Frontier-targeted data in Rich-like environments with exact argument and provenance supervision; do not prioritize answer-only supervision from these results. This is a diagnostic recommendation, not authorization to generate or train.
