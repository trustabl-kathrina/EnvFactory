# Graph-Targeted DPO Pilot

## Status

**PIPELINE-INVALID. Training and evaluation were not started.**

The preregistered data gate failed before optimizer initialization:

- Graph preference train contains 14 clean pairs, below the required deterministic 16-pair smoke.
- Only 8 independent `task_id + normalized environment state` frontiers exist after 672 Dynamic-v1 rollouts.
- Task-grouped preference-val consumes one task, leaving 7 train tasks and 14 pairs under the two-rejected-per-state cap.
- The requested 20 human-readable Graph examples cannot be produced; only 16 valid pairs exist in total.
- The smallest 16-pair Standard smoke has 71,029 preference tokens versus 41,275 for all 14 Graph train pairs, so budget comparability also fails.

No checkpoint exists for either preference arm. Consequently Frozen300, heldout60, depth breakdown, paired statistics, and BFCL were not run.

## Research questions

1. **Does ordinary trajectory DPO outperform Dynamic-v1?** Not answerable; Standard DPO was not trained because the paired experiment gate failed.
2. **Does Graph-localized DPO reduce `consumer_not_executed` more than Standard DPO?** Not answerable; no valid matched checkpoints were produced.
3. **Should Graph-targeted preference supervision be scaled up?** No under the current dataset/protocol. A larger independent EnvFactory TRAIN pool with at least 10 clean frontier tasks is needed before repeating the 16-pair smoke and 20-pair human audit.

## Final verdict

`PIPELINE-INVALID`

This verdict is about data sufficiency and comparison validity, not evidence that DPO is ineffective.

```text
NO GIT COMMIT
NO GIT PUSH
NO FULL TRAINING
```

