# Graph-Frontier Guided RL v1 readiness

Status date: 2026-09-17 (Asia/Singapore)

Decision: smoke gates are sufficient to begin the formal EnvFactory generation
stage.  Formal model training remains gated on conversion/contamination audit,
the materialized common prompt schedule, and both 128-prompt pilots.

## A. Existing code path

The audited live path is:

1. `src/graph/tool_graph.py::ToolGraph.build_tool_graph()`
2. `src/graph/sampler.py::TopologySampler`, wrapped by
   `graph_frontier/traceable_sampler.py::sample_with_dependency_trace()`
3. `src/gen/query_gen/query_gen_non_conv.py::QueryGenNonConv.gen()`
4. generation-time callback
   `graph_frontier/gold_sidecar.py::GenerationSidecarCallback.before_save()`
5. `src/utils/data_process.py::load_tool_chains()` and
   `convert_to_rl_data()`
6. `graph_frontier/guided_rl_generate.py::enrich_official_rows()` and
   `convert_and_audit()`
7. `graph_frontier/prepare_generated_guided_rl.py::prepare()`
8. `verl-agent/agent_system/environments/env_manager.py::make_envs()`
9. `verl-agent/.../envfactory/official_envs.py` reset/step/typed trace
10. `TrajectoryCollector.multi_turn_loop()`
11. `graph_frontier/guided_rl_reward.py::terminal_reward()`
12. unmodified VERL GRPO/FSDP trainer.

Generation-time graph metadata is preserved before
`ToolQueryNode.save()` discards `raw_tool_call` information.  No graph is
reconstructed from flattened SFT text.

## B. Generated RL data

Validated smoke pool:

- source environments: 88 existing executable MCP servers;
- graph: 683 tools, 4,633 parameters, 11,067 edges;
- repaired live graph SHA256:
  `9174edcf22221fad008bb3d112de135963760b1abf4494d4b86b2486bec77ea0`;
- unknown `user_provided` labels: 32 before repair, 0 after repair;
- generator: `guided_rl_generate.py generate`, seed `20260917`,
  Qwen2.5-14B-Instruct;
- smoke result: 14 unique executable tasks, train 12 / RL-val 2;
- depth distribution: 0=6, 1=4, 2=1, unknown=3;
- internal dependency edges: 10;
- unique query/tool-chain ratio: 1.0;
- environments represented: 13;
- frozen manifest SHA256:
  `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`;
- frozen exact overlap: 0.

The new dual-server generation smoke completed 2/2 chains with zero failure;
GPU0 and GPU1 each served four chat-completion requests and each produced one
raw chain plus one gold sidecar.

The formal 1,000-chain generation is running in tmux session
`gf_rl_v1_gen1000`, output
`repro_1p7b/data/graph_frontier_rl_v1_generated/formal1000_seed20260917`.
The expected yield is about 1,000--1,200 unique tasks.  Conversion must still
enforce 90--95% train split, generation-seed isolation, unique ratio, and zero
frozen overlap before training.

`NO RELEASED ENVFACTORY-RL USED AS FORMAL TRAINING DATA`

`FROZEN_300_TRAIN_OVERLAP = 0`

## C. Real graph sidecar sample

Task `gf-rl-v1-s1902130860-t0`, environment `BestBuyServer`, depth 1:

```text
BestBuyServer-search_products
  output products.sku (integer, internal)
    -> BestBuyServer-get_product_reviews
       input sku (integer, required)
```

The sidecar records stable tool/parameter IDs, tool sequence, seed, query,
initial/final scenarios, required/optional markers and task-level
`probe_eligibility`.  This sample has all eligibility checks true.  Tasks such
as the audited cross-turn Didi edge have `probe_eligibility.eligible=false` and
are forced into the broad pool with graph reward delta exactly zero.

## D. VERL integration

- `make_envs()` selects the EnvFactory adapter.
- Every rollout receives a private FastMCP client/process and mutable state.
- Real G=4 isolation test passed; one rollout's mutation did not change the
  other three states.
- Tool name, JSON arguments, returned fields, execution success/error and state
  snapshots are stored as typed rollout JSON, not reparsed from prose.
- Non-terminal step reward is zero.  The sole full-trajectory reward is emitted
  at termination, matching the collector's accumulated-reward semantics.
- Standard mode returns only `R_official`; Graph mode returns
  `R_official + 0.30*propagation + 0.40*consumer_completion - 0.10*invalid_retry`.
  Semantic success is not duplicated.
- Only explicit internal edges from a task whose task-level eligibility is
  exactly true can contribute graph reward.

## E. Two-GPU profile

Verified full-parameter Qwen3-1.7B/FSDP configuration:

- GPUs: 2 x A100 40 GiB;
- BF16, gradient checkpointing on, LoRA rank 0, critic off;
- train batch 2, per-GPU microbatch 1;
- rollout TP=2, vLLM memory utilization 0.30;
- prompt 5,200, response 768, max model length 6,144;
- G=4 resource run: 8 typed rollouts, 1 optimizer update, 117.0 s;
- GPU0 peak 29,896 MiB, GPU1 peak 29,896 MiB;
- headroom 11,064 MiB per GPU;
- trainer allocated 28.714 GiB, reserved 37.559 GiB;
- exit code 0, GPUs returned to 0 MiB afterward.

The G=4 random resource batch had reward variance but no completed graph edge;
it is a resource gate, not the functional graph gate.

## F. Standard RL smoke

PASS on the real two-task gate, G=2, one optimizer update:

- global step 1;
- grad norm 2.754;
- policy loss -0.061;
- KL loss 0.001;
- official reward min/mean/max: 0.0 / 0.375 / 0.5;
- 8 typed events;
- peak memory 29,896 MiB on both GPUs.

## G. Graph RL smoke

PASS on exactly the same data/seed/config/prompt schedule; only reward mode
changed:

- global step 1;
- grad norm 0.183;
- policy loss 0.037;
- KL loss 0.001;
- direct graph reward mean 0.700;
- nonzero propagation rollouts: 1;
- nonzero completion rollouts: 3;
- 8 typed events;
- peak memory 29,994 MiB on both GPUs.

Synthetic fixed sanity decomposition:

| trajectory | R_official | propagation | completion | retry fraction | graph delta | total |
|---|---:|---:|---:|---:|---:|---:|
| A correct | 1.00 | 1.00 | 1.00 | 0.00 | +0.70 | 1.70 |
| B producer, no consumer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| C consumer, wrong value | 0.00 | 0.00 | 1.00 | 0.00 | +0.40 | 0.40 |
| D unchanged invalid retry | 0.00 | 0.00 | 0.00 | 0.50 | -0.05 | -0.05 |

Thus A > B/C/D.  C retains partial completion credit; B intentionally gets no
completion bonus because the primary target is executing the consumer.

## H. Frozen formal configs

Both experiments use the same original Dynamic v1 initialization, generated
dataset, materialized prompt order, seed, G=4 and optimizer budget.  Frontier
sampling is materialized once and shared by A/B; the only training difference
is reward mode.

Common pilot:

- 128 prompt samples, 65/35 targeted/broad, 512 rollouts, 64 optimizer steps;
- same 2-GPU full-parameter config as section E;
- no frozen-300 evaluation.

Common formal:

- 1,000 materialized prompt samples, 65/35 targeted/broad;
- 4,000 rollouts, 500 optimizer steps;
- same prompt schedule hash and parquet hashes for Standard and Graph;
- independent checkpoint directories;
- both start from the original Dynamic v1, never from one another.

Experiment A uses `ENVFACTORY_REWARD_MODE=official`.  Experiment B uses
`ENVFACTORY_REWARD_MODE=graph_frontier`.  Formal launch is fail-fast on data
audit or either pilot gate.  A one-step G=4 measurement extrapolates to roughly
16--18 hours per 500-step run, excluding checkpoint and evaluation overhead.

## I. Remote changes and source-control boundary

Guided-RL additions/updates:

- `graph_frontier/guided_rl_generate.py`
- `graph_frontier/guided_rl_reward.py`
- `graph_frontier/prepare_generated_guided_rl.py`
- `graph_frontier/real_env_smoke.py`
- `graph_frontier/run_guided_rl_generation.sh`
- `graph_frontier/run_guided_rl_smoke.sh`
- `graph_frontier/run_guided_rl_train.sh`
- `graph_frontier/summarize_guided_rl_smoke.py`
- four guided-RL CPU test modules
- verl-agent EnvFactory adapter and its six tests
- compatibility patch records under `repro_1p7b/patches/`.

Tests at this gate:

- local guided-RL suite: 15/15 passed;
- remote main-repo suite: 13/13 passed;
- remote verl-agent adapter suite: 6/6 passed.

Unrelated pre-existing Dynamic-v2 and confirm-300 files remain untouched.  No
runtime logs or checkpoints are staged.

`NO GIT COMMIT`

`NO GIT PUSH`
