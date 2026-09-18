# Breeze TTS 2（C++/GGUF）能力与集成记录

> 目的：换掉 AuK，改用单一 TTS 后端 `Breeze-TTS-2.cpp`。本文件先记录**模型底层能力**与**预期新能力**，
> 再记录测试与迁移，避免清理 AuK 时丢失信息。项目约定见根目录 `AGENTS.md`。

## 1. 底层能力（以官方 README / docs / 源码为准）

单一 24 kHz 双语（中/英）模型，四种生成模式，由「有无参考音 / 有无指令」决定：

| 模式 | 输入 | 得到 | 内部判定（`apps/server/server.cpp`） |
| --- | --- | --- | --- |
| Voice Design | 文本 + 音色描述 | 凭空造声 | 无 `ref_audio` |
| Voice Clone | 参考音 + **逐字转写** + 文本 | 同音色念新文本 | 有 `ref_audio`，未发 `instruction` |
| Voice Direction | 参考音 + 转写 + **指令** + 文本 | 同音色，按指令改表演 | 有 `ref_audio`，且发了 `instruction` |
| Voice Conversion（实验） | 原始录音 + 目标音色 | 保留节奏/重音只换音色 | `POST /v1/audio/convert` |

关键实现事实（读源码得到，**不是**猜测）：

- `has_ref = !ref.codes.empty() && !ref.text.empty()`（`src/generation.cpp:95`）。
  **参考音没有转写就等于没有参考**，会退化成 design。所以克隆必须保证 `ref_text` 非空。
- 条件分支文本固定为 `spk + <ins_bos> + instruction + <ins_eos> + text`（`generation.cpp:40`），
  即指令始终进入 cond 分支；`cfg_scale>1` 时才同时算 uncond 再做 CFG。
- Vocal events 是文本内联标签：英文 `(laugh)/(sigh)/(cough)/(clears throat)`，中文 `[笑]/[叹气]/[咳嗽]/[清嗓子]`；
  词表开放（`(whispering)` 等也行）。**默认 `cfg 1.0` 基本不触发，需要 `2~3`**，>3 会发糙。
- 服务端**单并发**：第二个请求直接 `409 busy`，不排队。
- 合成响应是**无头 s16le mono 24kHz PCM**，不是 WAV；`X-Sample-Rate` 给采样率。
- 长文本服务端自动按句切分（`split_chars`，默认 600），逐段生成且以首段为条件保持音色。
- 保存音色 `--save-voice name` 缓存参考音编码（`.breeze`，约 800 B/s），首音 ~900ms → ~280ms；
  名字只允许 `[A-Za-z0-9_-]{1,64}`（会当文件名）。
- 采样默认（GGUF 元数据）：`temperature 0.9 / top_k 50 / top_p 1.0 / repetition_penalty 1.1 /
  depth_temperature 0.9 / depth_top_k 50`，`max_new_tokens 750`。
- 后端：本机用 CUDA 13.0（`/home/a2heng/下载/cuda13-toolkit`，sm_89）编译；上游实测 **Vulkan 比 CUDA 快**
  （本项目暂用 CUDA，Vulkan 需另装 `glslc`/`libvulkan-dev`）。

## 2. 已验证（真实跑通）

- 构建：`build/breeze-cpp/` 下 `breeze-cli`/`breeze-server`/`breeze-convert`/`breeze-quantize`；
  `ldd` 确认 `libcudart.so.13`/`libcublas.so.13` 经 RUNPATH 解析，运行无需 `LD_LIBRARY_PATH`。
- 模型：`ckpts/Breeze-TTS-2.cpp/breeze-tts-2-q8_0.gguf`（Q8_0，3.3 GB，**唯一选定版本**）+ `ref_voice.wav`。
- CLI 冒烟：voice design 与 voice clone 均成功，`backend: CUDA0`，输出 `build/breeze-cpp/smoke/`。
- HTTP：`breeze-server` 带 `--webui` 起在 `127.0.0.1:8137`，`/health` 返回
  `{"status":"ok","sample_rate":24000,"ws_port":8138}`；curl 克隆请求 200、149760 bytes PCM。
- 参考音必须是服务端 `read_wav_buffer` 能解析的格式；**AuK 旧参考音是 32-bit float**（`subtype="FLOAT"`），
  上传前需统一转 **PCM16 WAV**（见下）。

## 3. 预期新能力（相对 AuK，换模型的意义）

1. **可指令化的表演**：克隆音色 + 自然语言 direction（情绪/语速/语气），音色不被破坏。
   AuK 克隆无法用自由指令控制情绪（见 `docs/audiobook-workflow.md` §7.1）。
2. **Vocal events**：文本内联 `[叹气]`/`[笑]` 等非语言声（需 `cfg 2~3`）。
3. **更强指令跟随的造声**：角色参考音可由 description 现场"制造"。
4. **Voice conversion**（实验）：保留原表演换音色。
5. **流式低延迟**：HTTP/WebSocket，可做实时/对话。

## 4. 目标流水线（用户确定）

```
制造声音（voice design 造角色参考音，逐字转写用设计样本文本本身）
  → 克隆（固定参考音 + ref_text）
  → 表演指导（按句方向指令，需要时叠加 vocal events）
```

最终章节/整书默认输出 **MP3 128k**（逐行 wav 为无损缓存）。

LLM（Qwen3.8-27B + froggeric 修正模板）负责在标注/渲染阶段产出「谁在说 + 每句表演方向」；TTS 只用 Breeze。

## 5. 测试记录

- `tests/test_textnorm.py`：既有 6 项通过。
- `tests/test_breeze_renderer.py`：新增，覆盖
  - 参考音无转写时**不得**以 design 模式上传（fail-closed）；
  - 多部分表单编码、响应 PCM 解码、错误保留原因（离线、无网络/模型）。
- Lint：`ruff check` / `ruff check --select I` / `ruff format --check`（line-length 130，py310）。

## 6. 迁移清理（AuK → Breeze）

删除：`third_party/AuK` 子模块、`app/`、`vendor/` shims、`run.py`、`audiobook/renderer.py`、
`audiobook/voicebank.py`、`audiobook/instructions.py`、`scripts/build_voicebank.py`、
`scripts/optimize_refs.py`、`scripts/download_models.py`，及 AuK 权重
`ckpts/AuK`、`ckpts/AuK-Flash`、`ckpts/Qwen2.5-Omni-3B`（释放 ~24 G）。

保留：`audiobook/` 前端（cleaning/marks/schema/textnorm/llm/stats/canonical/live）+ 单文件 `audiobook/tts.py`、
`scripts/`（run_book/mark_script/marks_to_script/render_book/serve_breeze/serve_files/prepare_refs/
serve_llm*）、`ckpts/llm`（gemma 等，全部保留）、`ckpts/Breeze-TTS-2.cpp`、`third_party/breeze-tts`
（PyTorch 参考实现，pristine）、`third_party/Breeze-TTS-2.cpp`。
