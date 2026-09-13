#!/usr/bin/env bash
# Launch the local OpenAI-compatible llama.cpp server for audiobook preprocessing.
#
# Tuned on RTX 4070 Ti SUPER 16GB (see docs/audiobook-workflow.md):
#   n_gpu_layers=65 (full offload), n_ctx=16384, KV cache q8_0, flash_attn.
#   n_ctx=32768 OOMs at full offload on this card.
#
# Env overrides:
#   AUDIOBOOK_LLM_GGUF  model path (default ckpts/llm/Qwen3.8-27B-UD-IQ4_XS.gguf)
#   AUDIOBOOK_LLM_PORT  listen port (default 8080)
#   AUDIOBOOK_LLM_NGL   n_gpu_layers (default 65)
#   AUDIOBOOK_LLM_CTX   context size (default 16384)
#   AUDIOBOOK_LLM_KV    KV cache quant ggml type int (default 8 = q8_0)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# llama.cpp CUDA build needs the CUDA 13 runtime bundled with the venv (torch cu130)
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

MODEL="${AUDIOBOOK_LLM_GGUF:-$ROOT/ckpts/llm/Qwen3.8-27B-UD-IQ4_XS.gguf}"
PORT="${AUDIOBOOK_LLM_PORT:-8080}"
NGL="${AUDIOBOOK_LLM_NGL:-65}"
CTX="${AUDIOBOOK_LLM_CTX:-16384}"
KV="${AUDIOBOOK_LLM_KV:-8}"

exec "$ROOT/.venv/bin/python" -m llama_cpp.server \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --n_gpu_layers "$NGL" --n_ctx "$CTX" \
  --n_batch 2048 --n_ubatch 512 \
  --flash_attn True \
  --type_k "$KV" --type_v "$KV" \
  --verbose False
