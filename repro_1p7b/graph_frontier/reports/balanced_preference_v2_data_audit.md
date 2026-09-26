# Balanced Preference v2 data audit

Verdict: DATA_READY; unique train states: 58; pairs: 64.
A/B/C: {'continue_required': 29, 'downstream_continue': 22, 'stop_required': 13}; chosen tool/final ratios: 0.7969/0.2031.
Chosen valid: 1.0000; TRAIN-validation task overlap: 0; Frozen300 overlap: 0.
Validation A/B/C: {'continue_required': 17, 'downstream_continue': 20, 'stop_required': 1}; promoted whole TRAIN task: ['gf-rich-c8f65d5abe7142a34fce'].
State pair cap: 2; observed failure types: {'downstream_stop': 11, 'downstream_wrong_tool': 11, 'extra_tool_after_completion': 11, 'premature_stop': 26, 'useless_retry': 2, 'wrong_tool': 3}.

Caveat: only eight train stop states produced real extra-tool behavior at K=8; one such task was moved wholly into preference-validation before training.
The independent 21-task/29-frontier/10-strict-stop heldout tasks were not used in preference construction.
