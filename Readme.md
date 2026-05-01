One-line status
===============

This challenge closed today, and while I learned alot, and will continue experimenting, I had difficulties aligning my local repo with the online infra used for the challenge, and the challenge period expired before I had meaningful access.  This was an oversight on my part and a learning experience in using runpod as a cloud host.  I was granted additional credits and so I will continue to work on this project on my own initiative.  

Take - aways
============

The training schedules for the Jetson Orin Nano were especially illuminating, not in a leaderboard sense, but it exposed some power / heat contraints I hadn't seen in running small models on the Jetson previously.  Training steps were handled quite well for the small memory envelope and I wished multiple times that I had opted for the 16gb version instead of the 8gb version as it likely could have handled the training near as well as my 4070 powered PC and likely better than my base model m4 mac mini.

At 8gb the constraints were real though, both on hardware and memory, running headless will be my next trial as I acheived stable results within the memory envelope available ... the heavy workload on validation however lead to the need to schedule power modes along with training steps so as not to trigger power / heat surges on validation and model creation... not a concern I had with any of the other hardware I tried.  The logs I captured using tegra stats and the nvidia power utility will likely prove useful as I continue exploring the constraints on mobile robotics powered by local ai models and their real-world constraints.

One objective was ensuring tracability throughout experimentation, as I am delving into llm model creation for the first time:  taking notes, building tools and dashboards was a large focus, and exposing the workings a main driver.  Some of the the most valuable artifacts from this set of experiments were the dashboards and scripts developed to test and monitor the process, on windows / wsl / jetpack ubuntu on nvidia hardware / macos.

Branchpoint Summary
===================

- TODO: Document Jetson Branch (power /heat)
- TODO: Document MacOS Branch (metal.?)
- TODO: Document SP8192 tokenization / new tokenizer, sliding eval, bigramhash, and other techniques trialed and take-aways
- TODO: Document csv driven stage scheduler and env / txt / md / csv evolution
- TODO: Upload collected data analysis once completed. All branches.

This scratch branch is now a generic trainer/control-surface branch, not just a competition baseline.
It exposes the important training, architecture, optimizer, schedule, data, and export surfaces as env-driven controls,
with early validation, log visibility, manifest output, and dashboard schema coverage.

What Exists Now
===============

1. Run / manifest / provenance

- `RUN_ID`
- `MANIFEST_PATH`
- `DASHBOARD_PROFILE_NAME`
- manifest written beside logs by default
- manifest includes:
  - resolved env
  - model shape
  - optimizer groups
  - control tensor init config
  - LR multiplier config
  - stage config
  - data surface config
  - quant/export policy
  - export estimates
  - git commit / branch

2. Data surface

- `DATA_PATH`
- `TOKENIZER_PATH`
- `TRAIN_SHARD_LIMIT`
- `TRAIN_SHARD_OFFSET`
- `TRAIN_SHARD_ORDER_FILE`
- `TRAIN_SHARD_LIST`

Default training data behavior is still sequential streaming through resolved shards.
The loader now supports explicit subset and ordering controls without changing default behavior.

3. Evaluation surface

- `EVAL_POLICY=auto|chunked|sliding`
- `EVAL_STRIDE`
- `EVAL_MAX_TOKENS`
- `EVAL_SUBSET_MODE=full|head|tail`
- `EVAL_STRICT_FULL`

This supports both strict full validation and fast local probe evaluation.

4. Training schedule / curriculum

- `ITERATIONS`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`
- `LR_WARMUP_STEPS`
- `MAX_WALLCLOCK_SECONDS`
- `TARGET_GLOBAL_ACCUM`
- single-switch curriculum:
  - `BATCH_TOKENS_START`
  - `BATCH_SCHEDULE_FRACTION`
  - `SEQ_LEN_START`
  - `SEQ_SCHEDULE_FRACTION`

5. Generic multi-stage scheduler

- `STAGE_FRACTIONS`
- `STAGE_TRAIN_BATCH_TOKENS`
- `STAGE_TRAIN_SEQ_LEN`
- `STAGE_EMA_ENABLED`
- `STAGE_EMA_DECAY`
- `STAGE_TOK_LR_MUL`
- `STAGE_HEAD_LR_MUL`
- `STAGE_MATRIX_LR_MUL`
- `STAGE_SCALAR_LR_MUL`
- `STAGE_MUON_MOMENTUM`
- `STAGE_BIGRAM_SCALE`

Stage 0 is applied as the real live starting state.
Stage switches are logged during training.

6. Model shape / structure

- `VOCAB_SIZE`
- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `TIE_EMBEDDINGS`
- `TIED_EMBED_INIT_STD`
- `LOGIT_SOFTCAP`
- `ROPE_BASE`
- `ROPE_DIMS`
- `QK_GAIN_INIT`

Structural modules exposed:

- `BIGRAM_VOCAB_SIZE`
- `BIGRAM_DIM`
- `BIGRAM_SCALE_INIT`
- `USE_SMEARGATE`
- `SMEAR_GATE_INIT`
- `XSA_LAST_N`
- `XSA_LAYER_MASK`

7. Control-tensor init surface

- `ATTN_SCALE_INIT`
- `ATTN_SCALE_INIT_BY_LAYER`
- `MLP_SCALE_INIT`
- `MLP_SCALE_INIT_BY_LAYER`
- `RESID_MIX_INIT`
- `RESID_MIX_INIT_BY_LAYER`
- `Q_GAIN_INIT_BY_LAYER`
- `SKIP_WEIGHT_INIT`
- `SKIP_WEIGHT_INIT_BY_INDEX`

Resolved values are applied before compile, logged, and stored in the manifest.

8. Optimizer surface

Base groups already split as:

- token embeddings
- optional untied lm head
- matrix params on Muon
- scalar/control params on Adam

Base knobs:

- `EMBED_LR`
- `HEAD_LR`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `BETA1`
- `BETA2`
- `ADAM_EPS`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `GRAD_CLIP_NORM`
- `EMA_ENABLED`
- `EMA_DECAY`

Role LR multipliers:

- `TOK_LR_MUL`
- `HEAD_LR_MUL`
- `MATRIX_LR_MUL`
- `SCALAR_LR_MUL`

Layer/family LR multipliers:

- `ATTN_Q_LR_MUL_BY_LAYER`
- `ATTN_K_LR_MUL_BY_LAYER`
- `ATTN_V_LR_MUL_BY_LAYER`
- `ATTN_PROJ_LR_MUL_BY_LAYER`
- `MLP_FC_LR_MUL_BY_LAYER`
- `MLP_PROJ_LR_MUL_BY_LAYER`
- `CONTROL_LR_MUL_BY_LAYER`

9. Export / quant surface

Profiles:

- `EXPORT_PROFILE=safe|balanced|aggressive|custom`

Generic export knobs:

- `DEFAULT_ROLE_BITS`
- `DEFAULT_EMBED_BITS`
- `EMBEDDING_BITS`
- `BIGRAM_BITS`
- `MLP_FC_BITS`
- `MLP_PROJ_BITS`
- `ATTN_Q_BITS`
- `ATTN_K_BITS`
- `ATTN_V_BITS`
- `ATTN_PROJ_BITS`
- `OTHER_BITS`

Per-layer export bit overrides:

- `MLP_FC_BITS_BY_LAYER`
- `MLP_PROJ_BITS_BY_LAYER`
- `ATTN_Q_BITS_BY_LAYER`
- `ATTN_K_BITS_BY_LAYER`
- `ATTN_V_BITS_BY_LAYER`
- `ATTN_PROJ_BITS_BY_LAYER`

Per-layer export overrides now allow `float` entries.

Float passthrough controls:

- `CONTROL_TENSOR_NAME_PATTERNS`
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`
- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `INT8_KEEP_FLOAT_STORE_DTYPE`

Clip controls:

- `INT8_CLIP_PERCENTILE`
- `EMBEDDING_CLIP_PERCENTILE`
- `BIGRAM_CLIP_PERCENTILE`
- `MLP_FC_CLIP_PERCENTILE`
- `MLP_PROJ_CLIP_PERCENTILE`
- `ATTN_Q_CLIP_PERCENTILE`
- `ATTN_K_CLIP_PERCENTILE`
- `ATTN_V_CLIP_PERCENTILE`
- `ATTN_PROJ_CLIP_PERCENTILE`
- `OTHER_CLIP_PERCENTILE`

Export reporting now includes:

- baseline tensor bytes
- current payload bytes
- estimated packed bytes
- corrected packed raw estimate
- corrected packed compressed heuristic

Important Status Notes
======================

1. This is still a readable launch-point trainer, but it is now close to the upper edge of “single-file healthy”.
   For production import, parsing/schedule/export helpers should likely move out to modules.

2. The exporter still stores sub-8-bit values inside int8 tensors.
   That means:
   - current artifact sizes are real for the current serializer
   - packed-size numbers are estimates, not a true bitpacked artifact size

3. The branch now supports:
   - strict/full runs
   - local probe runs
   - curriculum experiments
   - structure experiments
   - export-policy experiments
   without requiring code edits for each test

4. Dashboard schema is now expected to track every new env surface added to the runtime.
   That rule should be kept during clean-branch import.

Suggested Clean-Branch Port Order
=================================

1. Port parser utilities + manifest writer
2. Port data/eval surface
3. Port control-tensor init surface
4. Port optimizer role/layer multiplier surface
5. Port stage scheduler
6. Port structural module surfaces
7. Port export surface + estimates
8. Then apply only the chosen winning/personal-best policies

Freeze Readiness
================

This branch is now suitable to archive as:

- scratch / learning / lab-surface branch

Then re-enter cleanly from a fresh repo pull for:

- best-run branch
- competition-targeted branch
- app-import branch

# Fleming 9 Training Course Model

A coherent training-plan draft based on the Fleming 9 / Parameter Golf metaphor.

## Core rules

The objective is **par**, achieved with three simple rules:

1. **Par 4 holes** use **shot 1 for setup** and **shot 2 to reach the green**.
2. **Par 3 holes** try to **reach the green in shot 1**.
3. **Every hole is finished with 2 putts**.

This creates a compact control grammar:

- **Par 3** = `on green` + `putt 1` + `putt 2`
- **Par 4** = `setup` + `on green` + `putt 1` + `putt 2`

## Layer / hole ordering

Handicap-transposed layer order:

- 1-based: `7, 5, 1, 9, 6, 2, 3, 4, 8`
- 0-based: `6, 4, 0, 8, 5, 1, 2, 3, 7`

Pars in that order:

`4, 3, 4, 3, 3, 3, 3, 3, 4` = **Par 30**

Blue-course hole lengths in that order:

`425, 235, 405, 165, 200, 140, 195, 140, 260` would be the strict scorecard order.

For the working training course below, the hole sequence used is:

`405, 140, 195, 140, 235, 200, 425, 260, 165`

Total: **2165 steps**

## Reach / play assumptions

Useful abstractions:

- **250-ish** is near maximum full tee-shot reach.
- **235** is the safer control threshold for deterministic planning.
- Some holes are reachable in one by raw distance, but geometry still makes the correct play a setup shot.
- Six holes are effectively **straight/direct**.
- Three holes are **setup-first** or geometry-sensitive.

## Club / shot abstraction

Starter bag:

- **D** = Driver
- **3** = Long iron / driving-iron class
- **9** = Short iron / lofted precision club
- **P** = Putter

Likely expanded bag:

- **D, 3, 6, 9, P**

Shot qualities:

- **Line-drive** = flatter, longer, more roll
- **Flyball / lofted** = higher, softer landing, less roll
- **Putt 1** = rim finder / prep - get within 3 ft
- **Putt 2** = hole shot - close hole

## Warmup idea

Before the course:

- regular stretch / compile warmup
- bucket-at-the-range style `LR_WARMUP_STEPS` such as **35 or 50**
- objective is to calm the early step-2 spike before scoring begins

---

# 2165-Step Course

## Hole 1 — 405 steps — h3 — long hinge with setup — Par 4

**Intent:** long setup hole with a meaningful hinge.

- Shot 1 — **255** — all-out driver / setup push
- Shot 2 — **145** — mid-iron style approach to green edge
- Shot 3 — **4** — putt 1, get within 3 ft
- Shot 4 — **1** — putt 2, finish

Cumulative end: **405**

## Hole 2 — 140 steps — h6 — straight technical hole — Par 3

**Intent:** direct attack, little drama.

- Shot 1 — **135** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **545**

## Hole 3 — 195 steps — h7 — medium direct hole — Par 3

**Intent:** long iron / heavy controlled drive to green.

- Shot 1 — **190** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **740**

## Hole 4 — 140 steps — h8 — short direct hole — Par 3

**Intent:** short drive / mid iron / heavy short iron to green.

- Shot 1 — **135** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **880**

## Hole 5 — 235 steps — h2 — threshold direct hole — Par 3

**Intent:** control-threshold hole; reachable in one under the planning grammar.

- Shot 1 — **230** — regular drive / heavy long iron to green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **1115**

## Hole 6 — 200 steps — h5 — straight carry hole — Par 3

**Intent:** 80% drive / long iron to green.

- Shot 1 — **195** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **1315**

## Hole 7 — 425 steps — h1 — long setup hole — Par 4

**Intent:** long fairway setup, then approach over protected landing.

- Shot 1 — **255** — fairway setup / hinge
- Shot 2 — **165** — long-iron or heavy mid-iron approach to green
- Shot 3 — **4** — putt 1
- Shot 4 — **1** — putt 2

Cumulative end: **1740**

## Hole 8 — 260 steps — h9 — dogleg / chip setup hole — Par 4

**Intent:** positional shot first, then short chip to green.

- Shot 1 — **235** — setup drive to position
- Shot 2 — **20** — short-iron / chip to green
- Shot 3 — **4** — putt 1
- Shot 4 — **1** — putt 2

Cumulative end: **2000**

## Hole 9 — 165 steps — h4 — short lofted finisher — Par 3

**Intent:** short straight shot over trees; loft and stick.

- Shot 1 — **160** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **2165**

---

# 10× Course — 21650 Steps

Same hole order, same ratios, every shot length multiplied by **10**.

## Hole 1 — 4050 steps — h3 — Par 4

- Shot 1 — **2550**
- Shot 2 — **1450**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **4050**

## Hole 2 — 1400 steps — h6 — Par 3

- Shot 1 — **1350**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **5450**

## Hole 3 — 1950 steps — h7 — Par 3

- Shot 1 — **1900**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **7400**

## Hole 4 — 1400 steps — h8 — Par 3

- Shot 1 — **1350**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **8800**

## Hole 5 — 2350 steps — h2 — Par 3

- Shot 1 — **2300**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **11150**

## Hole 6 — 2000 steps — h5 — Par 3

- Shot 1 — **1950**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **13150**

## Hole 7 — 4250 steps — h1 — Par 4

- Shot 1 — **2550**
- Shot 2 — **1650**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **17400**

## Hole 8 — 2600 steps — h9 — Par 4

- Shot 1 — **2350**
- Shot 2 — **200**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **20000**

## Hole 9 — 1650 steps — h4 — Par 3

- Shot 1 — **1600**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **21650**

---

## Notes

- This document keeps the golf-course abstraction coherent without forcing a literal implementation.
- The 2165-step version is the working compact course.
- The 21650-step version is the same course at 10× scale, preserving hole and shot ratios exactly.
- This is best treated as a **training grammar / control scaffold**, not yet as the final competition branch.
