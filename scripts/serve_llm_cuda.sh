#!/usr/bin/env bash
# Launch the compiled CUDA llama.cpp server (with native MTP) for audiobook preprocessing.
#
# Default model is Ornith-1.5-9B: qwen35 hybrid arch, Q8_0 (~9.8 GB), native MTP.
# Tuned for RTX 4070 Ti SUPER 16GB: full offload (-ngl 99), ctx 32768, q8_0 KV,
# draft-mtp depth 3 (measured draft acceptance 0.78-0.93 -> ~1.6x decode).
#
# Env overrides:
#   AUDIOBOOK_LLM_GGUF   model path (default ckpts/llm/Ornith-1.5-9B-Q8_0.gguf)
#   AUDIOBOOK_LLM_PORT   default 8080
#   AUDIOBOOK_LLM_NGL    default 99
#   AUDIOBOOK_LLM_CTX    default 32768
#   AUDIOBOOK_LLM_KV     q8_0|q4_0|f16 ... default q8_0
#   AUDIOBOOK_LLM_THINK_BUDGET  default 512 (-1 unlimited, 0 off)
#   AUDIOBOOK_LLM_SPEC   default "draft-mtp" (empty to disable)
#   AUDIOBOOK_LLAMA_BIN / AUDIOBOOK_CUDA_TOOLKIT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${AUDIOBOOK_LLAMA_BIN:-/home/a2heng/下载/llama.cpp/build/bin}"
TK="${AUDIOBOOK_CUDA_TOOLKIT:-/home/a2heng/下载/cuda13-toolkit}"
export LD_LIBRARY_PATH="$BIN:$TK/lib64:${LD_LIBRARY_PATH:-}"

MODEL="${AUDIOBOOK_LLM_GGUF:-$ROOT/ckpts/llm/Ornith-1.5-9B-Q8_0.gguf}"
PORT="${AUDIOBOOK_LLM_PORT:-8080}"
NGL="${AUDIOBOOK_LLM_NGL:-99}"
CTX="${AUDIOBOOK_LLM_CTX:-32768}"
KV="${AUDIOBOOK_LLM_KV:-q8_0}"
BUDGET="${AUDIOBOOK_LLM_THINK_BUDGET:-512}"
SPEC="${AUDIOBOOK_LLM_SPEC-draft-mtp}"

DRAFT="${AUDIOBOOK_LLM_DRAFT:-}"
NM=${AUDIOBOOK_LLM_SPEC_DRAFT_N_MAX:-3}
NCMOE="${AUDIOBOOK_LLM_N_CPU_MOE:-}"          # keep MoE experts of the first N layers on CPU
BATCH="${AUDIOBOOK_LLM_B:-2048}"
UBATCH="${AUDIOBOOK_LLM_UB:-512}"
moe_args=()
if [ -n "$NCMOE" ]; then moe_args=(-ncmoe "$NCMOE"); fi
spec_args=()
if [ -n "$SPEC" ]; then
  spec_args=(--spec-type "$SPEC" --spec-draft-n-max "$NM" --spec-draft-type-k "$KV" --spec-draft-type-v "$KV")
fi
draft_args=()
if [ -n "$DRAFT" ]; then
  draft_args=(-md "$DRAFT")
fi

exec "$BIN/llama-server" \
  -m "$MODEL" --host 127.0.0.1 --port "$PORT" --jinja \
  -np 1 --ctx-checkpoints 0 --no-cache-idle-slots \
  --reasoning-format deepseek --reasoning-budget "$BUDGET" \
  -ngl "$NGL" -c "$CTX" -b "$BATCH" -ub "$UBATCH" \
  -ctk "$KV" -ctv "$KV" \
  "${moe_args[@]}" "${draft_args[@]}" "${spec_args[@]}"
