# AGENTS.md

## 项目速览（外层 wrapper 结构）

- **AuK 是子模块**：`third_party/AuK`（官方 `Tencent-Hunyuan/AuK`，**保持 pristine，一行不改**）。
- **我们的代码在外层**：`app/`（`infer_gradio.py`、`audio_io.py`、`template_helper.py`、`patches.py`）、`vendor/`（shims）、`run.py`、`scripts/`、`requirements.txt`、`AGENTS.md`。
- **shims（`vendor/`）**：`torchaudio.py`（→ `app.audio_io`）、`silero_vad.py` / `funasr.py`（→ faster-whisper）。让未改动的 AuK 直接 `import torchaudio` / PE 的 `silero_vad`、`funasr` 也能跑。
- **patches（`app/patches.py`）**：`apply_patches()` 注入 Qwen 8-bit 量化、DiT 转 bf16，替代原先对 AuK 的修改。
- 运行环境：外层 `.venv`，依赖见 `requirements.txt`（钉死版本）。
- 启动 WebUI：`source .venv/bin/activate && python run.py`（`run.py` 设置 `sys.path`：`vendor` → `third_party/AuK/src` → 外层根）。
- 直接跑源码：`PYTHONPATH=third_party/AuK/src:. .venv/bin/python ...`（无需 `pip install -e .`）。
- ckpts 与 `assets/voice-reference` 在外层；demo 音频在子模块 `third_party/AuK/assets`（`app.infer_gradio.submodule_path`）。
- **有声书流水线（`audiobook/` + `scripts/`）**：小说 TXT → 舞台剧本 → `script.csv` → 逐段渲染 → 母带。详见 `docs/audiobook-workflow.md`。
  - **完整流程（唯一入口）**：`scripts/run_book.py <txt> --book <name>`，stages=`prepare → script → convert → render`（可续跑，自动起停本地 LLM）：
    1. **prepare**（`build_book.py --prepare-only`）：清洗/规范化/切章 → `outputs/<book>/{source,clean}.txt` + `chapters/chNNN.txt`。
    2. **script**（**`scripts/mark_script.py`，唯一路径**）：逐章 → 舞台剧台本；**不做频率发现、不预置人名表**，人物词典在逐章标注中**增量维护**（见下）。
    3. **convert**（`scripts/marks_to_script.py --marked-dir`）：`⦃角色␟内容⦄` 机械解析（无 LLM）→ `script.csv`/`script.json`；cast 缺省时由 `script/roles.json` 回退构造（声线留到 TTS 阶段）。
    4. **render**（`scripts/render_book.py`）：AuK `zero_shot_tts` 逐行合成 + 拼接 → `outputs/<book>/render/`。
  - **标记约定**：台词 = `⦃角色名␟朗读内容⦄`（罕见符号 U+2983/U+241F/U+2984，可用 `AUDIOBOOK_MARK_OPEN/CLOSE/SEP` 覆盖），标记外一律旁白；解析/渲染在 `audiobook/marks.py`。
  - **剧本标注（唯一路径，`scripts/mark_script.py`）**：
    - 模型只当「阅读文本→舞台剧台本」的编剧，**只有一个 `edit` 工具**（`op=speak/delete/replace`），逐处一次调用、定位片段 ≤6 字；**去引号、删多余「名字：」归因、气口处加逗号都由 MCP 代码机械执行**（不做语义判断）。
    - **一对多人物词典**（规范名 → 称呼/绰号/代称等标签），**不要路人**，也**不预置/不带入固定人名表**（冷启动为空，只从 `script/roles.json` 续跑）；**每章开头先 `maintain_roster` 做人物提取/挖掘**维护词典，`speak` 的 role 一律写规范名，标注后把标签归一为规范名；每章末落一次 `roles.json`。次要人物不在此路人化，留到 TTS 阶段做。
    - **滚动压缩（`--mode seq`）**：始终只维护**一条逻辑对话**。每章上下文 = `system`（编剧提示词 + 人物词典）+ `user`（**前情摘要** + 本章正文）；章内跑完整 tool-loop，**章末调用 `compress()`**：用「上一版摘要 + 本章出场角色（`parse_marks`）+ 本章正文」让 LLM 产出新的**滚动摘要（≤400 字，写 `summary.txt`）**，只保留对判断「谁在说话」有用的信息（新人物、身份、关系、称呼变化、剧情要点）。下一章只带两样耐久记忆：**人物词典**与**前情摘要**，上一章正文与全部工具调用丢弃。
    - **补漏**：章末 `unmarked_quotes` 找出未处理引号，回炉重标直到 0。
    - MCP 原语（仅 `set_text`/`edit`/`get_marked`）在 `scripts/script_mcp_server.py`，客户端 `audiobook/mcp.py`。
    - 产物：`outputs/<book>/script/{chNNN.marked.txt, roles.json, summary.txt, *.html(段落表+diff)}`。
  - **预处理**：`cleaning`（编码/引号/去页码）+ `textnorm.clean_for_llm`（**通用字符白名单**：只留 L/N/P/M 类别，LLM 前生效）。站点广告/元数据在输入 txt 层一次性删除。
  - **适用范围**：目前只在一部中文网文（本地示例语料）上验证过；换小说需重写「人物脚本」提示词（`scripts/mark_script.py` 的 `SYSTEM/ROSTER_SYSTEM` 示例），其它阶段通用。频率发现/固定人名表（`finalize_roster`/`namefinder`/`roster`）与旧 LLM 抽取/断句/合并路径（`extract`/`segment`/`agent`/`pipeline`/`merge_roster`/`merge_narration`）已删除。
  - 后端：`voicebank`（instruct 造参考音 → whisper ASR → 克隆）、`renderer`（AuK `zero_shot_tts` 逐行，可续跑）、`assembler`（拼接 + -14 LUFS）；CLI `scripts/build_voicebank.py`、`scripts/render_book.py`。参考音准备：`scripts/prepare_refs.py`（24kHz 单声道、≤12s、去静音）。
  - 输出统一在 `outputs/<book>/`；LLM 缓存 `.cache/llm/`；两者均已 gitignore。
  - 局域网浏览/试听：`scripts/serve_files.py`，systemd `auk-files.service`（`:8899`）。

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
   - **自检**：正确脱离后，子进程应为 `fd0=/dev/null`、`fd1=fd2=日志文件`、`ppid=1`、独立 `sid`。用 `pgrep -x <comm>` 精确定位进程；**别用 `pgrep -f <pattern>`**，它会匹配到 bash -c 包装进程自身（或当前命令行），导致看错对象。（`watch nvidia-smi` 那种是用户自己终端里的常驻命令，不是卡死，先分清。）
2. **所有可能联网/加载大模型的命令显式加 `timeout <秒>`**，禁止给单条命令设置几十分钟的超时。
3. **凡导入 torch / transformers / gradio 或触发模型下载，统一带上环境变量**：
   ```
   GRADIO_ANALYTICS_ENABLED=False \
   HF_HUB_DISABLE_TELEMETRY=1 \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \   # 仅在模型已缓存时
   HF_ENDPOINT=https://hf-mirror.com \          # 需要联网下载时用国内镜像
   NO_PROXY='*' PYTHONPATH=src .venv/bin/python ...
   ```
4. **不要在自动化检查里调用 `build_demo()` / `demo.launch()`**：会触发 Gradio 遥测 HTTP 请求并可能卡住。只对纯函数写检查；UI 构造如需验证，用最小 `Blocks` 并禁用遥测。
5. **模型（faster-whisper Whisper/VAD、Qwen2.5-Omni）预先下载到本地缓存**，不要让运行时懒下载；参考 `scripts/download_models.py`。
6. **pip 安装**用后台 + 日志轮询，装完必须 `pip check`。
7. 长任务结束/中断后，**清理遗留后台进程**（`jobs`/`kill`）。
8. **`pkill -f <pattern>` 要小心**：pattern 会匹配到当前这条命令行自身，可能把正在执行的 shell 一起杀掉；先 `pgrep -af` 看精确 PID 再 kill。

## 经验/踩坑（持续补充）

- **判断「卡死」先分清性质**：多是 opencode 在等后台进程交还 stdin 管道（见规约 1），不是进程真死；先在另一个命令里 `pgrep -af` / `ps` 核实。
- **hf-mirror 下载**：`huggingface_hub` 直连 `https://hf-mirror.com` 可用（免代理）；`HF_ENDPOINT` 在**进程启动时**读取，改了要重启脚本。大文件用 `hf_hub_download`（自带 `.incomplete` 断点续传），失败可重试续传，别删 `.incomplete`。
- **GGUF 只下主权重**：`imatrix_*.gguf` 仅在量化时用、`mmproj-*.gguf` 是视觉投影、`config.json`/`README` 是 Hub 元数据，llama.cpp 推理都不需要。
- **不要在正在写入的大文件上跑 `find`/`grep`/`ls -R`**：会放大 I/O 等待，看起来像卡住。
- **AuK 克隆无法用自由指令控制情绪/风格**（实测 2026-09-13）：往 `zero_shot_tts` 指令里加"terrified/angry"等描述，模型会把指令前缀当台词念出来再接文本（ASR 可证）。克隆只能控音色（参考音）+ 时长（`gen_seconds`）；情绪/语速/音调走 `emotion_edit`/`speed_edit`/`pitch_edit` 后编辑。见 `docs/audiobook-workflow.md` §7.1、实验记录 `docs/prep-experiments.md`（复现脚本已清理删除）。
- **参考音质量是克隆天花板**（实测 2026-09-13）：`assets/voice-reference/逗哥音色整理合集` 是 TTS 合成音色，克隆会放大其瑕疵（多余停顿/生硬）；VAD 去静音 + bwe + 收紧 `gen_seconds` 都无法根治。要自然必须换更高质量参考音。见 docs §6.4。

## 验证与代码风格

- Lint/格式（对齐 `.pre-commit-config.yaml`，ruff 固定 `0.11.2`）：
  ```
  .venv/bin/ruff check <files>
  .venv/bin/ruff check --select I <files>
  .venv/bin/ruff format --check <files>
  ```
  需要格式化时用 `ruff format <files>`。
- 不要提交密钥；`.env` 不进版本库。
- 除非用户明确要求，不要提交 git commit。
- **所有构建/渲染输出一律写在项目内 `outputs/`**（如 `outputs/<book>/`），不要写 `/tmp`；`outputs/` 与 `.cache/` 已在 `.gitignore`。

## 关键约定

- 模型下载**默认走魔搭 ModelScope**：`python scripts/download_models.py [--model auk-flash]`，或 `modelscope download --model ... --local_dir ./ckpts/...`；HuggingFace 仅为备选。`AuK-Flash` 是 4 步蒸馏版（对应 WebUI 的 `AuK-Flash ⚡`，`ckpts/AuK-Flash`），`AuK` 是 base 版。
- 音频 I/O 与重采样统一走 `app/audio_io.py`（soundfile + scipy）；AuK 侧靠 `vendor/torchaudio.py` shim 满足，**不安装/不使用真 torchaudio**。
- 本地 ASR/VAD 统一走 faster-whisper：AuK 通过 `vendor/funasr.py`、`vendor/silero_vad.py` shim 间接使用，不引入 funasr/modelscope/silero-vad。
- PyTorch 使用 CUDA 13.0 轮子（NJU 镜像，`+cu130`）；8-bit 量化需 `accelerate` + `bitsandbytes`。
- Prompt Enhancer 需要外部 LLM API Key；WebUI 默认关闭。
- **不要修改 `third_party/AuK` 内的任何文件**；需要改行为时放到 `app/`（新模块或 patches）。

## 任务/功能总览（来源：`third_party/AuK/src/auk/infer/pe.config.yaml` + `pe.py`）

PE 的作用：把「用户口语指令 +（可选）音频」分类到唯一 task，抽取结构化参数，机械渲染成标准 instruction，并算好 `gen_seconds`（目标时长）。16 个任务如下；`输入` 中「音频」= `needs_audio`，「文本」= `needs_text`。

| task | 中文 | 输入 | 关键参数 | 时长策略 | 输出目的 |
| --- | --- | --- | --- | --- | --- |
| `instruct_tts` | 文本合成 | 文本（无音频） | `style_desc` 风格描述、`text` 要念文本 | `tts_by_text` | 按文字描述的音色/风格从零合成，音色随机 |
| `zero_shot_tts` | 音色克隆 | 音频 + 文本 | `text` | `tts_by_text` | 用参考音频的音色念指定文本 |
| `content_edit` | 内容编辑 | 音频 | subtype∈`insert_before/insert_after/delete/delete_before/delete_after/replace`；`anchor`/`text`/`target`/`orig`/`new` | `content_scaled` | 改「说了什么」，音色/其它不变 |
| `vocal_edit` | 歌词编辑 | 音频（歌唱） | `orig`、`new` | `content_scaled` | 改歌词，旋律/唱腔/音色不变 |
| `speed_edit` | 语速调整 | 音频 | `speed_multiplier`∈`0.5/0.75/1.25/1.5/2.0` | `speed_scaled` | 调语速，内容/音色不变 |
| `volume_edit` | 音量调整 | 音频 | subtype∈`increase/decrease`；`gain_db`∈`5/10/15` | `equal_length` | 调大/调小音量 |
| `pitch_edit` | 音调调整 | 音频 | subtype∈`increase/decrease`；`semitones`∈`1/2/3` | `equal_length` | 升/降音调（半音） |
| `emotion_edit` | 情感转换 | 音频 | `emotion`∈`happy/angry/sad/fearful/surprised/disgusted/calm/excited` | `equal_length` × 情感系数 | 转情感，内容/音色不变 |
| `voice_edit` | 音色编辑 | 音频 | `timbre_desc` 音色描述 | `equal_length` | 保留原话内容，只换音色 |
| `accent_edit` | 去口音 | 音频 | 无 | `equal_length` | 去方言口音，发音标准化，音色一致 |
| `nonverbal_edit` | 非语言声编辑 | 音频 | subtype∈`delete/add_after/add_before/add_head/add_tail`；`sound`、`anchor` | `equal_length` ± 非语言增量 | 增/删呼吸、笑、咳、叹气等非语言声 |
| `whisper_edit` | 耳语转换 | 音频 | subtype∈`to_normal/to_whisper` | `to_normal` 用原始时长，其余 `equal_length` | 耳语↔正常发声 |
| `enhance_speech` | 语音增强 | 音频 | `cleanup_mode`(可选)∈`denoise/dereverb/denoise_dereverb` | `equal_length`（跳过 VAD） | 去噪+去混响，保留所有说话人 |
| `separate_speech` | 说话人分离 | 音频 | subtype∈`by_content/by_order`；`text` 或 `n`；`cleanup_mode`(可选) | `equal_length`（跳过 VAD） | 多说话人里只留指定人 |
| `extract_vocals` | 提取人声 | 音频 | subtype∈`singing_only/all_human_voices` | `equal_length`（跳过 VAD） | 只保留歌声 / 保留全部人声，去伴奏 |
| `improve_quality` | 音质提升 | 音频 | subtype∈`bandwidth_extension/remove_effect`；`effect`(remove_effect)、`cleanup_mode`(可选) | `equal_length`（跳过 VAD） | 带宽扩展、去电话/扩音器音色、恢复清晰度 |

### 时长确定规则（`third_party/AuK/src/auk/infer/pe.py::_compute_duration`，PE 侧）

- `base_duration`：先做 VAD 裁剪后的「有效语音时长」；若 task ∈ `vad.skip_tasks`（`enhance_speech/separate_speech/extract_vocals/improve_quality/nonverbal_edit/vocal_edit`）或 `whisper_edit/to_normal`，则用原始音频时长。
- `speed_scaled`：`base_duration / speed_multiplier`。
- `content_scaled`：`base_duration × edited_spoken / original_spoken`；有 ASR 原文时按「原文 + 增删内容」估；无 ASR 时 `replace` 用 `new/orig` 时长比，否则回退 `base_duration`。
- `tts_by_text`：F5 文本时长估算（`seconds_per_utf8_byte`：en 0.0656 / zh 0.0803；`<10` 字节短文本 speed 0.3）；`instruct_tts` 可由 LLM 在 F5 基线上微调，`zero_shot_tts` 若有 ASR 参考文本则 `base × 目标/参考 权重比`。
- `equal_length`：等于 `base_duration`；`emotion_edit` 再乘情感系数（sad 1.22、fearful 1.16、其余 1.06）；`nonverbal_edit` 再加减增量（呼吸 +0.35/-0.6；咂嘴/吸鼻/惊讶等 +0.5/-1.0；笑/叹气/咳/清嗓 +0.75/-1.05；默认 +0.55/-0.9，结果下限 0.1s）。
- 最终按 `output_frames_per_second=50` 量化到模型 latent 帧对齐；用户显式 `target_duration>0` 时优先用用户值。

### 前置能力与降级

- 任务分类、`instruct_tts` 风格扩写、`voice_edit` 音色扩写依赖外部 LLM（`LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_NAME`）；无 Key 时 PE 不可用。
- ASR：优先腾讯云录音文件识别（`TENCENTCLOUD_SECRET_ID/KEY`），失败或未配置时走本地 `faster-whisper`（通过 `vendor/funasr.py` shim 接入 AuK 的 PE；默认 `small`/`int8`/CPU，可用 `AUK_ASR_MODEL/AUK_ASR_DEVICE/AUK_ASR_COMPUTE_TYPE` 覆盖）。
- VAD：`vendor/silero_vad.py` shim → `faster_whisper.vad`（Silero VAD via onnxruntime）。
- WebUI 默认关闭 PE；关闭时 `Duration=0` 由 `app/infer_gradio.py::estimate_duration` 本地粗算：有音频按音频时长，否则按「文本字符/词 + 标点停顿」估算（强停顿 `。！？!?…` 0.30s，弱停顿 `，、；：,;:` 0.12s，中文 0.22s/字、英文 0.40s/词）。
- Duration 下方有「时长倍率」按钮（`1.0`–`2.0`，步进 `0.1`，`DURATION_RATES`），对自动估算结果再乘倍率；生成时 `effective = estimate_duration(...) * rate`。
- 合成类指令（`_TTS_HINT_RE`：zero-shot/instruct TTS，含「same voice/音色克隆/生成语音内容/用…声音说」）**即使加载了参考音频也按目标文本估时长**，不取参考音频长度；仅音频编辑类任务用源音频时长。
- 上限：自动估算 `MAX_AUTO_DURATION=300s`（不再卡 30s），手动滑杆 `MAX_MANUAL_DURATION=120s`；`gen_seconds` 会按上限外的显式值传给模型（模型配置 `max_duration=65536` 帧，ComfyUI 侧另有 source+target≤30s 约束）。
- 长文本分句合成：合成类任务若文本估算超过 `MAX_SEGMENT_SECONDS=20s`，`run_generate` 会按标点分句（`_split_text_for_duration`，过长再按弱标点/定长切），逐句生成（复用同一参考音频）后拼接（句间插 `SEGMENT_GAP_SECONDS=0.15s` 静音），避免单次生成爆显存；每句时长按文本占比分摊总时长。
