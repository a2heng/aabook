# 有声书工作流（TXT → 舞台剧本 → 音频）

把一部小说 TXT 变成有声书：清洗切章 → 逐章标注成舞台剧台本 → 机械转成 `script.csv` → AuK 逐段合成 → 母带。

## 唯一入口

```bash
source .venv/bin/activate
python scripts/run_book.py assets/txt/novel.txt --book dawn
# 只跑某段：
python scripts/run_book.py assets/txt/novel.txt --book dawn --stages script
```

stages（可续跑，自动起停本地 LLM）：`prepare → script → convert → render`

## LLM 服务（标注用）

- 默认 `ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf`（unsloth Dynamic 2.0，内置 MTP）：`scripts/serve_llm_cuda.sh` 默认 `--spec-type draft-mtp --spec-draft-n-max 6`；采样 temp 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5；用模型自带 chat template（不要另传 jinja）。
- 可选 DFlash 提速：`z-lab/Qwen3.5-9B-DFlash` 用 `convert_hf_to_gguf.py --target-model-dir ckpts/llm/Qwen3.5-9B-hf --outtype bf16` 转 GGUF，再 `AUDIOBOOK_LLM_SPEC=draft-dflash AUDIOBOOK_LLM_DRAFT=<dflash.gguf>`。9B 无 DSpark 草稿。
- 环境变量覆盖见 `scripts/serve_llm_cuda.sh` 头部注释；LLM 与 Breeze（8137）不能同时占满 GPU。

| 阶段 | 命令 | 产物 |
| --- | --- | --- |
| prepare | `build_book.py <txt> --out outputs/<book>` | `source.txt`、`clean.txt`、`chapters/chNNN.txt`、`chapters.json` |
| script | `mark_script.py <first> --count N --book <book> --mode seq` | `script/chNNN.marked.txt`、`roles.json`、`summary.txt`、`*.html` |
| convert | `marks_to_script.py --marked-dir outputs/<book>/script --out outputs/<book>` | `script.csv`、`script.json` |
| render | `render_book.py --script outputs/<book>/script.csv ...` | `render/{rows,chapters,book.wav}` |

预处理：`cleaning`（编码/引号/去页码；方括号只去 `[` `]` 符号、**不删括号里的字**）→ `textnorm.clean_for_llm`（通用字符白名单 + 标点规范化，**省略号统一变 `。`**，LLM 前生效）。站点广告/元数据在输入 TXT 层先删掉。

## 标记约定

台词写成 `⦃角色名␟朗读内容⦄`；标记外一律旁白。符号是罕见字符（U+2983 / U+241F / U+2984），可用 `AUDIOBOOK_MARK_OPEN/CLOSE/SEP` 覆盖。解析/渲染在 `audiobook/marks.py`（无 LLM）。

## 剧本标注（`scripts/mark_script.py`，唯一路径）

模型只当「阅读文本 → 舞台剧台本」的编剧，**只有一个 `edit` 工具**；所有机械改动由 MCP 代码执行：

- `edit(op="speak", text="“带引号的整段”", role="规范名")`：去引号、包成 `⦃…⦄`，并删掉引号前多余的 `名字：` 归因。
- `edit(op="delete", text="“词”")`：去引号留词；引号内只有标点（`“…”`）则整段删；单独标点直接删。
- `edit(op="replace", find, replace)`：补句末标点；**气口（换气/停顿处）加逗号**。
- 定位片段要求 ≤6 字；模型抄整句会失败，失败后自动回炉（见下）。

### 一对多人物词典

- 词典是 `规范名 → 标签`（正式名、称呼、绰号、代称…），**不用路人**，也不预置固定人名表。
- **每章开头** `maintain_roster`：LLM 从本章正文挖出标签并并入词典；`speak` 的 role 一律写规范名，标注后把标签归一为规范名。每章末落一次 `roles.json`。
- 次要人物不在此路人化；留到 TTS 阶段按需处理。

### 一步一条逻辑对话 + 滚动压缩

始终只维护一条逻辑对话。每章上下文 = `system`（编剧提示词 + 人物词典）+ `user`（**前情摘要** + 本章正文）：

1. 章内跑完整 tool-loop（模型反复调用 `edit`）。
2. 章末 `compress()`：用「上一版摘要 + 本章出场角色（`parse_marks`）+ 本章正文」让 LLM 产出新的**滚动摘要**（≤400 字，写 `summary.txt`），只保留判断「谁在说话」需要的信息。
3. 下一章只带两样耐久记忆：**人物词典**与**前情摘要**；上一章正文和全部工具调用丢弃。
4. 补漏：章末 `unmarked_quotes` 找出未处理引号，回炉重标直到 0。

### 实时查看

`mark_script` 边跑边写 `script/live.jsonl`，并生成自刷新页 `script/live.html`：

```
http://<host>:8899/outputs/<book>/script/live.html
```

页面按聊天气泡展示：人物挖掘、每次 `edit` 调用与结果、每章摘要。

**原始 LLM 输入/输出**（监督用）：`script/llm_raw.html`（请求的每条 message 可展开、原始 reasoning/content/tool_calls，1.5s 自刷新）与 `script/llm_raw.jsonl`（逐条 JSONL，含 roster 与标注两类调用）；每次运行开始会清空。

## 产物

```
outputs/<book>/
  source.txt clean.txt chapters/            # prepare
  script/chNNN.marked.txt roles.json summary.txt live.jsonl live.html *.html
  script.csv script.json                    # convert
  render/                                   # render
```

## 备注

- AuK 克隆只能控音色与时长，情绪/语速/音调走 `emotion_edit`/`speed_edit`/`pitch_edit` 后编辑；参考音质量是克隆天花板。
- 换小说需重写 `mark_script.py` 的 `SYSTEM/ROSTER_SYSTEM` 示例；其余阶段通用。
- 旧的频率发现 / 固定人名表 / LLM 抽取断句路径已删除（负结果见 `docs/prep-experiments.md`）。
