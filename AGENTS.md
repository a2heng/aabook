# AGENTS.md

## 项目速览

- **唯一 TTS = Breeze TTS 2**：子模块 `third_party/Breeze-TTS-2.cpp`（C++/GGUF，含嵌套 `third_party/ggml`）+ 参考实现 `third_party/breeze-tts`。**两者均保持 pristine，一行不改**。
- **唯一 LLM = Qwen3.5-9B（MTP）**：本地 llama.cpp（CUDA）起 OpenAI 兼容服务，用于剧本标注；权重 `ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf`（unsloth Dynamic 2.0，~5.9 GB，**内置 MTP head**，`--spec-type draft-mtp --spec-draft-n-max 6`）。用**模型自带 chat template**（`scripts/serve_llm_cuda.sh` 默认不传模板；`qwen_chat_template.jinja`/froggeric 是 Qwen3.8 时代的修正模板，勿用于 3.5）；采样按官方 thinking：temp 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5。备选提速：`z-lab/Qwen3.5-9B-DFlash`（本地 `convert_hf_to_gguf.py --target-model-dir` 转 GGUF 后 `--spec-type draft-dflash`，无 9B DSpark）。`ckpts/llm` 其余模型（gemma-4-E4B、Qwen3.8-27B、Ornith、Spark）保留备选。
- **备选 LLM = Bonsai 27B（Ternary + DSpark）**：PrismML 的 llama.cpp fork 已做子模块 `third_party/llama.cpp`，编到 `build/llama-cpp/`（CUDA 参数同 Breeze）；权重只能用 fork 专用的 `ckpts/llm/Ternary-Bonsai-27B-PQ2_0.gguf`（上游 llama.cpp 用 `Q2_g64`，别混）；DSpark 草稿要先转：`gguf_dspark_to_dflash.py --drop-shared-tensors <legacy-Q4_1> <PQ2_0> <out-dflash.gguf>`。启动：`AUDIOBOOK_LLAMA_BIN=build/llama-cpp/bin` + `--spec-type draft-dspark --spec-draft-n-max 4` + KV q4_0；采样 temp 0.7 / top_p 0.95 / top_k 20。实测 98 tok/s（无草稿 62.6）@200W。
- **我们的代码在外层**：`audiobook/`（前端 + Breeze 渲染）、`scripts/`、`tests/`、`requirements.txt`、`AGENTS.md`。
- 运行环境：外层 `.venv`，依赖见 `requirements.txt`。Python 侧只做音频 I/O、ASR、响度、LLM 客户端；**不再需要 torch**。
- **目录规划**：`outputs/<book>/` 只放书产物；`build/breeze-cpp/` 放 Breeze 编译/试听产物；`benchmarks/` 放基准结果；`.cache/llm/`、`.cache/logs/` 放 LLM 缓存与服务日志。以上均已 gitignore。
- 旧 AuK 代码/子模块/权重（`app/`、`vendor/`、`run.py`、`audiobook/renderer.py`、`voicebank.py`、`instructions.py`、`ckpts/AuK*`、`ckpts/Qwen2.5-Omni-3B`）**已全部删除**，不要再引入。

## 有声书流水线（`audiobook/` + `scripts/`）

小说 TXT → 舞台剧本 → `script.csv` → 逐行渲染 → 母带。完整流程见 `docs/audiobook-workflow.md`。

- **唯一入口**：`scripts/run_book.py <txt> --book <name>`，stages=`prepare → script → convert → render`（可续跑，自动起停本地 LLM；render 前会停 LLM 腾 GPU）。
- **两段式工作流（固定）**：先**预热**前 10 章跑 `script`，人工**校订** `script/roles.json`（人物/别名）与 `script/voice_profiles.json`（性别/年龄），再**正式**跑全书（复用已校订的词典/声线）。预热示例：`run_book.py <txt> --book <name> --stages prepare,script --limit 10`。
  1. **prepare**（`scripts/build_book.py --prepare-only`）：清洗/规范化/切章 → `outputs/<book>/{source,clean}.txt` + `chapters/chNNN.txt`。
  2. **script**（`scripts/mark_script.py`）：逐章 → 舞台剧台本；**不做频率发现、不预置人名表**，人物词典在逐章标注中增量维护。
  3. **convert**（`scripts/marks_to_script.py --marked-dir`）：`⦃角色␟内容⦄` 机械解析（无 LLM）→ `script.csv`/`script.json`。
  4. **voicebank**（`scripts/build_voicebank_breeze.py`）：Breeze voice design 给每个角色造参考音 + 用设计样本文本作逐字转写 → `voicebank.json`/`voicebank_meta.json`。
  5. **render**（`scripts/render_book.py`）：**Breeze TTS 2** 逐行克隆/表演合成 + 拼接 → `outputs/<book>/render/`；最终章节/整书默认 **MP3 64k**（`--audio-format`，逐行 wav 仅作无损缓存）。
- **声线**：旁白**固化**在 `voices/narrator.json` + `narrator_m/f.wav`（git 追踪；`scripts/build_narrator.py` 复现；青年男/女，seed58，讲故事）。角色由 LLM 只定**性别+年龄**（`voice_profiles.json`），描述固定 `一位{年龄}{性别}性，日常说话，语气自然。`；同角色用确定性 seed（`role_seed`），设计 cfg=4、克隆 cfg=1。
- **标记约定**：台词 = `⦃角色名␟朗读内容⦄`（罕见符号 U+2983/U+241F/U+2984，可用 `AUDIOBOOK_MARK_OPEN/CLOSE/SEP` 覆盖），标记外一律旁白；解析在 `audiobook/marks.py`。
- **剧本标注（`mark_script.py`）**：
  - 模型只当「阅读文本→舞台剧台本」的编剧，**只有一个 `edit` 工具**（`op=speak/delete/replace`），逐处一次调用；去引号、删多余「名字：」归因、气口加逗号由 MCP 代码机械执行。
  - **一对多人物词典**（规范名 → **模糊名称 + 3~5 个具体称谓**），冷启动为空，只从 `script/roles.json` 续跑；每章开头 `maintain_roster`，`speak` 写规范名，章末落 `roles.json`。**绝不收指代/代词/描述性短语/泛称**（`MAX_ALIASES=5` 硬上限）。
  - **滚动压缩（`--mode seq`）**：每章上下文 = `system`（提示词 + 词典）+ `user`（前情摘要 + 本章正文）；章末 `compress()` 产 ≤400 字摘要写 `summary.txt`。下一章只带「词典 + 摘要」。
  - **补漏**：章末 `unmarked_quotes` 找出未处理引号，回炉重标直到 0。
  - MCP 原语（`set_text`/`edit`/`get_marked`）**内联在 `scripts/mark_script.py`**（`ScriptServer` + in-process `MCPClient`，无子进程）。
  - 产物：`outputs/<book>/script/{chNNN.marked.txt, roles.json, summary.txt, *.html}`。
- **预处理**：`cleaning`（编码/引号/去页码/**第一步去方括号——只去 `[` `]` 符号、保留括号内文字**）+ `textnorm.clean_for_llm`（只做字符白名单 L/N/P/M + 标点规范化；**省略号保留 `……`，ASCII `...` 归一为 `……`**）；**不做小句切分**，整章直接送 LLM。
- **Vocal events**：单一数据源 `audiobook/tts.py` 的 `VOCAL_EVENTS`（官方中文例子 `[笑]/[叹气]/[咳嗽]/[清嗓子]`，词表开放，`is_event_tag` 放行自由词）；标注 `speak` 可带 `tag`，程序写成 `[tag]` 前置；渲染器检测到 tag 自动把 cfg 提到 2.5。
- **速度**：Breeze 无时长/语速参数（自己决定停）；快慢用 direction 指令（`语速放慢`）、参考音语速，或后处理 `tts.atempo`（`row.speed` / `--speed`，保音高）。
- 后端**单文件** `audiobook/tts.py`：Breeze HTTP 渲染（可续跑）+ 拼接/响度（-16 LUFS）+ `encode_lossy`（MP3 64k）+ `atempo` 变速 + vocal events。参考音准备：`scripts/prepare_refs.py`。
- 局域网浏览/试听：`scripts/serve_files.py`（`:8899`）。

## Breeze TTS 2（C++/GGUF，唯一推理路径）

- **为什么用 C++ 版**：绕开 PyTorch 参考实现的依赖冲突（`qwen-tts` 钉 `transformers==4.57.3`/`torchaudio`/`sox`/`librosa`）。C++ 版只依赖 GGUF + CUDA/Vulkan。
- **权重**：`ckpts/Breeze-TTS-2.cpp/breeze-tts-2-q8_0.gguf`（**唯一选定版本**，3.3 GB，Q8_0；vocoder 已烘进 GGUF）+ 示例参考音 `ref_voice.wav`（转写：`The harbour lights came on one by one as the evening tide began to turn.`）。来源 `HoppouAI/Breeze-TTS-2.cpp`，用 `HF_ENDPOINT=https://hf-mirror.com hf download ...`。
- **构建（CUDA，复用本机 CUDA 13.0 toolkit）**：源码在外层编译，产物在 `build/breeze-cpp/`（gitignore）。工具链：`.venv/bin/cmake`/`ninja` + `/home/a2heng/下载/cuda13-toolkit`（rtx4070TiS sm_89）。
  ```
  PATH="$PWD/.venv/bin:/home/a2heng/下载/cuda13-toolkit/bin:$PATH" \
  .venv/bin/cmake -S third_party/Breeze-TTS-2.cpp -B build/breeze-cpp -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=OFF -DBREEZE_CUDA=ON \
    -DCUDAToolkit_ROOT=/home/a2heng/下载/cuda13-toolkit \
    -DCMAKE_CUDA_COMPILER=/home/a2heng/下载/cuda13-toolkit/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_RPATH=/home/a2heng/下载/cuda13-toolkit/lib64 \
    -DCMAKE_INSTALL_RPATH=/home/a2heng/下载/cuda13-toolkit/lib64 \
    "-DCMAKE_EXE_LINKER_FLAGS=-L/home/a2heng/下载/cuda13-toolkit/lib64 -Wl,-rpath-link,/home/a2heng/下载/cuda13-toolkit/lib64" \
    "-DCMAKE_SHARED_LINKER_FLAGS=-L/home/a2heng/下载/cuda13-toolkit/lib64 -Wl,-rpath-link,/home/a2heng/下载/cuda13-toolkit/lib64"
  PATH="$PWD/.venv/bin:/home/a2heng/下载/cuda13-toolkit/bin:$PATH" \
  .venv/bin/cmake --build build/breeze-cpp -j 12
  ```
  - **rpath/-rpath-link 必需**：toolkit `lib64` 不在默认搜索路径，否则链接期报 `libcudart.so.13/libcublas.so.13 not found`。已写进二进制 RUNPATH，运行无需 `LD_LIBRARY_PATH`。
- **服务**：`scripts/serve_breeze.py up|down|status`（默认 `127.0.0.1:8137`，WebUI 开）。`down` 只杀 pidfile 里校验过 comm 的自有进程。
- **产物**：`breeze-cli`（单次合成）、`breeze-server`（HTTP/WebSocket + WebUI，24kHz s16le PCM）、`breeze-convert`（声音转换，实验）、`breeze-quantize`、`libbreeze.so`。
- **四种模式**（由「有无参考音 / 有无指令」决定）：
  - **Voice Design 造声**：文本 + 描述，无参考音。
  - **Voice Clone 克隆**：参考音 + **逐字转写**（`ref_text` 必须非空，否则退化成 design）。
  - **Voice Direction 表演指导**：克隆 + `instruction`（情绪/语速/语气），音色不变。
  - **Voice Conversion 转换**：保留原表演换音色（`POST /v1/audio/convert`，实验）。
- **Vocal events**：文本内联 `[笑]/[叹气]/[咳嗽]/[清嗓子]`（英文 `(laugh)/(sigh)/(cough)/(clears throat)`），**需 `cfg 2~3` 才明显**，>3 发糙。
- **实测**（RTX 4070 Ti Super，CUDA，q8_0）：build/克隆/表演指导端到端可用，`backend: CUDA0`，RTF ≈0.7x。上游实测 **Vulkan 比 CUDA 快**；本项目暂用 CUDA，Vulkan 需另装 `glslc`/`libvulkan-dev`。

### Breeze 渲染接口（`audiobook/tts.py`）

- `BreezeRenderer.render_rows(rows, out_dir, voices=...)` → 逐行 wav，缓存按内容哈希，可续跑。
- 参考音经 `wav_bytes()` 统一转 **mono PCM16 WAV** 再上传（旧 32-bit float 参考音不可靠）。
- 克隆**失败即报错**（无转写时拒绝上传，不会静默变 design）；转写优先取 `voicebank_meta.json`，缺失才 ASR。
- 本地请求走无代理 opener；响应校验 PCM 长度与采样率。
- `--breeze-direction` 时把 `row.style_desc`/`row.emotion` 拼成中文 direction 指令；默认纯克隆。

## 命令执行规约（重要：避免「卡住」）

1. **禁止对长命令使用 `| head` / `| tail` 截断**。管道会缓冲输出，看起来像卡死。
   - 需要完整输出就直接输出（工具会把超长内容写入文件）。
   - 长时间任务改为后台 + 日志 + 轮询，且**必须完全脱离本会话**：
     `setsid --fork nohup cmd </dev/null > /tmp/opencode/xxx.log 2>&1 & disown`
     随后用 Read 读该日志，直到出现结束标志。
   - **根因**：opencode 用 `bash -c` 执行命令，**stdout/stderr 是 unix socket（不是 tty）**，工具一直读到该 socket EOF 才认为命令结束。任何后代进程只要还持有这个 socket，命令就永不「结束」→ 假死。
   - **两个必踩的坑**：
     1. `setsid` **必须加 `--fork`**。非交互 bash 无 job control，不加 `--fork` 时 `setsid` 直接 exec 成长任务，不会 daemonize。
     2. 别写 `cd X && setsid ... cmd &`：`&` 会把整个 `cd && setsid ...` 变成一个**子 shell 异步列表**，子 shell 会**等它的前台子进程**（即长任务），于是子 shell 攥着 socket 不放。应在后台命令前用 `;` 或先单独 `cd`。
   - **自检**：正确脱离后，子进程应为 `fd0=/dev/null`、`fd1=fd2=日志文件`、`ppid=1`、独立 `sid`。用 `pgrep -x <comm>` 精确定位进程；**别用 `pgrep -f <pattern>`**，它会匹配到 bash -c 包装进程自身（或当前命令行），导致看错对象。
2. **所有可能联网/加载大模型的命令显式加 `timeout <秒>`**，禁止给单条命令设置几十分钟的超时。
3. **模型/服务预下载到本地缓存**，运行时不要懒下载。需要联网下载时用 `HF_ENDPOINT=https://hf-mirror.com`、`NO_PROXY='*'`。
4. **不要在自动化检查里调用任何 `*.launch()` UI**：会触发遥测 HTTP 请求并可能卡住。只对纯函数写检查。
5. **pip 安装**用后台 + 日志轮询，装完必须 `pip check`。
6. 长任务结束/中断后，**清理遗留后台进程**（`jobs`/`kill`）。
7. **`pkill -f <pattern>` 要小心**：pattern 会匹配到当前这条命令行自身，可能把正在执行的 shell 一起杀掉；先 `pgrep -af` 看精确 PID 再 kill。优先 `pgrep -x`。

## 经验/踩坑（持续补充）

- **判断「卡死」先分清性质**：多是 opencode 在等后台进程交还 stdin 管道（见规约 1），不是进程真死；先在另一个命令里 `pgrep -af` / `ps` 核实。
- **hf-mirror 下载**：`huggingface_hub` 直连 `https://hf-mirror.com` 可用（免代理）；`HF_ENDPOINT` 在**进程启动时**读取。大文件用 `hf_hub_download`（自带 `.incomplete` 断点续传）。
- **GGUF 只下主权重**：`imatrix_*.gguf` 仅量化用、`mmproj-*.gguf` 是视觉投影、`config.json`/`README` 推理都不需要。
- **不要在正在写入的大文件上跑 `find`/`grep`/`ls -R`**：会放大 I/O 等待，看起来像卡住。
- **Breeze 克隆必须给转写**：源码 `has_ref = !codes.empty() && !text.empty()`，没有 `ref_text` 会退化成 design（音色完全不同）。`BreezeRenderer` 已 fail-closed。
- **参考音统一转 PCM16 WAV**：旧 AuK 参考音是 `subtype="FLOAT"`，上传前用 `wav_bytes()` 归一（mono PCM16 + ASCII 文件名）。
- **Breeze 服务单并发**：第二请求返回 `409 busy`；渲染必须串行。`health` 返回 `{status, sample_rate, ws_port}`。
- **Vocal events 要 `cfg 2~3`**，默认 1.0 基本不触发；cfg>3 声音发糙。
- **上游实测 CUDA 比 Vulkan 慢**（本机 RTF≈0.7x）；追求速度可再装 Vulkan SDK 重编。`-dd` 量化变体短句可、长文劣化，别用于 narration。
- **输出用 MP3 64k**：`tts.encode_lossy`（ffmpeg libmp3lame）把章节/整书 wav 转 `.mp3`，逐行 wav 仍是无损缓存；`--audio-format {mp3,aac,opus,wav}`、`--bitrate` 可切换。
- **造声参考音的转写用「设计样本文本」本身**，不要用 ASR（ASR 有错字，会静默劣化克隆）。
- **标注 LLM 必须用 froggeric 模板**：Qwen 官方 chat template 在工具调用上有空 think 毒化 / JSON 参数崩溃 / prefix KV 失效；`serve_llm_cuda.sh` 默认 `--chat-template-file assets/llm/qwen_chat_template.jinja --reasoning-format deepseek --reasoning-preserve`。实测 27B IQ4_XS 全量 GPU（14.7/16.4 GB，~35 tok/s），工具调用返回 `reasoning_content` + `tool_calls` 正常。LLM（8080）与 Breeze（8137）**不能同时占满 GPU**，render 前需停 LLM。

## 验证与代码风格

- Lint/格式（ruff，line-length 130，py310）：
  ```
  .venv/bin/ruff check <files>
  .venv/bin/ruff check --select I <files>
  .venv/bin/ruff format --check <files>
  ```
  需要格式化时用 `ruff format <files>`。
- 测试：`.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'`（离线纯函数，不加载模型/网络）。
- 不要提交密钥；`.env` 不进版本库。
- 除非用户明确要求，不要提交 git commit。
- **所有输出写在项目内**：书产物 `outputs/<book>/`，编译/试听 `build/`，基准 `benchmarks/`，缓存/日志 `.cache/`；不要写 `/tmp`；均已 gitignore。

## 关键约定

- **模型下载默认走魔搭 ModelScope**，HuggingFace/`hf-mirror` 为备选。
- 本地 ASR/VAD 统一走 faster-whisper（`faster_whisper.vad` 提供 Silero VAD），不引入 funasr/modelscope/silero-vad。
- 音频 I/O 与重采样统一走 soundfile + scipy；`prepare_refs.py` 做 24kHz 单声道 + VAD 去静音。
- **不要修改 `third_party/breeze-tts` 与 `third_party/Breeze-TTS-2.cpp` 内的任何文件**；需要改行为时放到外层 `audiobook/` 或 `scripts/`。
