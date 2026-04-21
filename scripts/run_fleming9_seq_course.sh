#!/usr/bin/env bash
set -euo pipefail

# Fleming 9 course curriculum for the 2165-step training plan in
# `fleming9_training_course.md`.
#
# Shot-to-seq mapping used here:
# - setup shots: 512
# - scoring / "driver" shots to green: 1024
# - short chip on hole 8: 128
# - putt 1: 32
# - putt 2: 4
#
# This wrapper only sets the curriculum surface. Other knobs can still be
# overridden from the environment before invoking it.

export ITERATIONS="${ITERATIONS:-2165}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"

# Exact optimizer-step starts for all 30 shots in the 2165-step course.
export STAGE_STEP_STARTS="${STAGE_STEP_STARTS:-0,255,400,404,405,540,544,545,735,739,740,875,879,880,1110,1114,1115,1310,1314,1315,1570,1735,1739,1740,1975,1995,1999,2000,2160,2164}"

# Per-shot sequence lengths:
# h1: 1024,512,32,4
# h2: 1024,32,4
# h3: 1024,32,4
# h4: 1024,32,4
# h5: 1024,32,4
# h6: 1024,32,4
# h7: 1024,512,32,4
# h8: 1024,128,32,4
# h9: 1024,32,4
export STAGE_TRAIN_SEQ_LEN="${STAGE_TRAIN_SEQ_LEN:-1024,512,32,4,1024,32,4,1024,32,4,1024,32,4,1024,32,4,1024,32,4,1024,512,32,4,1024,128,32,4,1024,32,4}"

# Per-shot batch tokens matched to sequence length:
# 1024 ->  524288
#  512 ->  262144
#  256 ->  131072
#  128 ->   65536
#   64 ->   32768
#   32 ->   16384
#   16 ->    8192
#    8 ->    4096
#    4 ->    2048
#    2 ->    1024
#    1 ->     512
export STAGE_TRAIN_BATCH_TOKENS="${STAGE_TRAIN_BATCH_TOKENS:-524288,262144,16384,2048,524288,16384,2048,524288,16384,2048,524288,16384,2048,524288,16384,2048,524288,16384,2048,524288,262144,16384,2048,524288,65536,16384,2048,524288,16384,2048}"

# Structured tracing is useful for this curriculum because the whole point is
# to inspect how training dynamics change at different shot lengths.
export STEP_TRACE_EVERY="${STEP_TRACE_EVERY:-1}"

# Variable-length stage curricula are much happier when compile is allowed to
# stay dynamic instead of fullgraph-specializing every new shape.
export COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
export DYNAMO_RECOMPILE_LIMIT="${DYNAMO_RECOMPILE_LIMIT:-64}"

if [[ $# -eq 0 ]]; then
  exec python train_gpt.py
fi

exec "$@"
