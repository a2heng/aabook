#!/usr/bin/env bash
# Launch the compiled CUDA llama.cpp server for audiobook preprocessing.
#
# Default model is Qwen3.5-9B MTP UD-Q4_K_XL (~5.9 GB) with the built-in MTP head for
# speculative decoding (--spec-type draft-mtp). Tuned for RTX 4070 Ti SUPER 16GB: full
# offload, ctx 32768. Qwen has its own chat template, so do NOT pass a jinja template.
#
# Env overrides:
#   AUDIOBOOK_LLM_GGUF       model path (default ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf)
#   AUDIOBOOK_LLM_TEMPLATE   jinja template (default "" ; set for Qwen3.8 / other runs)
#   AUDIOBOOK_LLM_PORT       default 8080
#   AUDIOBOOK_LLM_NGL        default 99
#   AUDIOBOOK_LLM_CTX        default 32768
#   AUDIOBOOK_LLM_KV         q8_0|q4_0|f16 ... default q8_0
#   AUDIOBOOK_LLM_FA         flash attention on|off|auto (default on; required for quantized KV)
#   AUDIOBOOK_LLM_THINK_BUDGET  default 512 (-1 unlimited, 0 off)
#   AUDIOBOOK_LLM_SPEC       default "draft-mtp" (empty disables)
#   AUDIOBOOK_LLM_SPEC_DRAFT_N_MAX  default 4 (Qwen3.5 MTP measured best; gemma-4 MTP head also 4)
#   AUDIOBOOK_LLM_DRAFT      extra draft gguf. Auto: gemma-4 MTP head. For DFlash/DSpark set
#                            SPEC=draft-dflash|draft-dspark and point this at the converted
#                            z-lab/Qwen3.5-9B-DFlash GGUF (see docs/audiobook-workflow.md).
#   AUDIOBOOK_LLAMA_BIN / AUDIOBOOK_CUDA_TOOLKIT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${AUDIOBOOK_LLAMA_BIN:-/home/a2heng/下载/llama.cpp/build/bin}"
TK="${AUDIOBOOK_CUDA_TOOLKIT:-/home/a2heng/下载/cuda13-toolkit}"
export LD_LIBRARY_PATH="$BIN:$TK/lib64:${LD_LIBRARY_PATH:-}"

MODEL="${AUDIOBOOK_LLM_GGUF:-$ROOT/ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
TEMPLATE="${AUDIOBOOK_LLM_TEMPLATE-}"
PORT="${AUDIOBOOK_LLM_PORT:-8080}"
NGL="${AUDIOBOOK_LLM_NGL:-99}"
CTX="${AUDIOBOOK_LLM_CTX:-32768}"
KV="${AUDIOBOOK_LLM_KV:-q8_0}"
FA="${AUDIOBOOK_LLM_FA:-on}"
BUDGET="${AUDIOBOOK_LLM_THINK_BUDGET:-512}"
SPEC="${AUDIOBOOK_LLM_SPEC-draft-mtp}"

DRAFT="${AUDIOBOOK_LLM_DRAFT-}"
NM="${AUDIOBOOK_LLM_SPEC_DRAFT_N_MAX:-}"
if [ -z "$DRAFT" ] && [[ "$MODEL" == *gemma-4* ]]; then
  DRAFT="$ROOT/ckpts/llm/MTP/mtp-gemma-4-E4B-it-Q8_0.gguf"
  [ -n "$NM" ] || NM=4
fi
[ -n "$NM" ] || NM=4
NCMOE="${AUDIOBOOK_LLM_N_CPU_MOE:-}"          # keep MoE experts of the first N layers on CPU
BATCH="${AUDIOBOOK_LLM_B:-2048}"
UBATCH="${AUDIOBOOK_LLM_UB:-512}"
moe_args=()
if [ -n "$NCMOE" ]; then moe_args=(-ncmoe "$NCMOE"); fi
spec_args=()
if [ -n "$SPEC" ]; then
  spec_args=(--spec-type "$SPEC" --spec-draft-n-max "$NM" -ngld "$NGL" --spec-draft-type-k "$KV" --spec-draft-type-v "$KV")
fi
draft_args=()
if [ -n "$DRAFT" ] && [ -f "$DRAFT" ]; then
  draft_args=(-md "$DRAFT")
fi
template_args=()
if [ -n "$TEMPLATE" ] && [ -f "$TEMPLATE" ]; then
  template_args=(--chat-template-file "$TEMPLATE")
fi

exec "$BIN/llama-server" \
  -m "$MODEL" --host 127.0.0.1 --port "$PORT" --jinja \
  -np 1 --ctx-checkpoints 0 --no-cache-idle-slots \
  --reasoning-format deepseek --reasoning-budget "$BUDGET" \
  -ngl "$NGL" -c "$CTX" -b "$BATCH" -ub "$UBATCH" -fa "$FA" \
  -ctk "$KV" -ctv "$KV" \
  "${template_args[@]}" \
  "${moe_args[@]}" "${draft_args[@]}" "${spec_args[@]}"
