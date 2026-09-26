# Balanced Conditional Transition Preference v2

Verdict: GO for a bounded mechanism pilot, not authorization for full training.
Mechanism gates: {'consumer_local_preserved': True, 'stop_shortcut_repaired': True, 'downstream_full_improved': True, 'overexecution_reduced': True, 'official_not_regressed': True}.
Train: 64 pairs / 58 states; A/B/C = {'continue_required': 29, 'downstream_continue': 22, 'stop_required': 13}.
Frozen300 overlap: 0; task split overlap: 0.
Frontier exact: 11/29; flow: 20/29; strict stop: 6/10.
Downstream exact: 6/12.
Official reward: 0.261905; success: 1/21.
Full metrics: {'final_answer': 6, '8_step_still_tool_calling': 4, 'adjacent_repeated_call_tasks': 2, 'total_tool_calls': 45, 'producer_reach': 9, 'producer_success': 9, 'consumer_opportunity': 9, 'consumer_success': 5, 'parameter_correct': 5, 'downstream_eligible': 3, 'post_consumer_continuation': 1, 'final_state_match': 8, 'length_penalty_mean': 0.05, 'trace_score_mean': 0.09523809523809523, 'state_score_mean': 0.5238095238095238, 'format_penalty_mean': 0.026190476190476188}

Limitations:
- 21 heldout tasks and one deterministic trajectory per task
- only one real stop-negative validation task; stop validation accuracy is high variance
- re-split was selected after observing available stop negatives but before training; heldout remained sealed
