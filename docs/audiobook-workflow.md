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

- 默认 `ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf`（unsloth Dynamic 2.0，内置 MTP）：`scripts/serve_llm_cuda.sh` 默认 `--spec-type draft-mtp`（草稿长度 `AUDIOBOOK_LLM_SPEC_DRAFT_N_MAX` 默认 4）；采样 temp 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5；用模型自带 chat template（不要另传 jinja）。
- 可选 DFlash 提速：`z-lab/Qwen3.5-9B-DFlash` 用 `convert_hf_to_gguf.py --target-model-dir ckpts/llm/Qwen3.5-9B-hf --outtype bf16` 转 GGUF，再 `AUDIOBOOK_LLM_SPEC=draft-dflash AUDIOBOOK_LLM_DRAFT=<dflash.gguf>`。9B 无 DSpark 草稿。
- 环境变量覆盖见 `scripts/serve_llm_cuda.sh` 头部注释；LLM 与 Breeze（8137）不能同时占满 GPU。
- **前缀缓存**：默认开启（`-np 1`，不要加 `--no-cache-idle-slots`/`--ctx-checkpoints 0`）；另加 `--cache-reuse 256`（`AUDIOBOOK_LLM_CACHE_REUSE`，0 关闭）复用跨章节的共同片段，避免每章重新 prefill。

| 阶段 | 命令 | 产物 |
| --- | --- | --- |
| prepare | `build_book.py <txt> --out outputs/<book>` | `source.txt`、`clean.txt`、`chapters/chNNN.txt`、`chapters.json` |
| script | `mark_script.py <first> --count N --book <book> --batch N` | `script/chNNN.marked.txt`、`roles.json`、`summary.txt`、`*.html` |
| convert | `marks_to_script.py --marked-dir outputs/<book>/script --out outputs/<book>` | `script.csv`、`script.json` |
| render | `render_book.py --script outputs/<book>/script.csv ...` | `render/{rows,chapters,book.wav}` |

预处理：`cleaning`（编码/引号/去页码；方括号只去 `[` `]` 符号、**不删括号里的字**）→ `textnorm.keep_layout`：**段落、首行缩进、全部标点原样保留**（`——`/引号/`※` 等不动），只去掉我们自用的 `[` `]` `<` `>` 定界符；ASCII `...` 归一为 `……`。站点广告/元数据在输入 TXT 层先删掉。live 页按段落渲染（`<p>` + 2em 首行缩进）。

## 标记约定

台词写成 `<角色名>朗读内容</角色名>`（`audiobook/marks.py` 的 `MARK_RE`，可用 `AUDIOBOOK_MARK_RE` 覆盖）；标记外一律旁白。vocal events 写在内容里，如 `<高文>[叹气]好吧。</高文>`。

## 剧本标注（`scripts/mark_script.py`，唯一路径）

模型只当「阅读文本 → 舞台剧台本」的编剧，工具只有 `edit`：`speak` 标注 / `unmark` 去误标；所有机械改动由 MCP 代码执行：

- 正文**不加任何标记/编号**，注入纯原文；`edit(op="speak", text="这段发言的完整原文", role="规范名")`：
  **text 必须逐字照抄完整发言**（引号标点一不差，禁止半截/改写/带旁白）。
- **MCP 定位（提示词不提）**：先精确匹配；**≤10 字绝不开模糊**；长于 10 字做**两次锚点匹配**
  （开头 10 字一次、结尾 10 字一次，中间允许差异，span=头锚到尾锚）。定位后自动裁到引号内；
  **跨多引号自动拆成「每段一个标记」**（旁白留在标记外）；半截引号自动配对；span 内旧标记撤掉重包
  （覆盖，无嵌套）；无引号 span 必须有「说道/想」类提示，否则按旁白拒绝；**原文一个字不改**。
- `edit(op="unmark", …)` 去掉误标（只去标记、不动原文）；同一处再 `speak` 会覆盖旧标记并支持收紧范围。
- **1 次标注 + 1 轮分窗检查（每窗 ~20 句；`--check-steps` 总步数默认 100，0=关闭）**：标完机械扫描
  **未被任何标记包住的引号内容**（`unmarked_quotes`，带 `[3B]~[3C]` 与出处片段）喂给模型定点补标；
  检查只改确有问题的，收尾打印 `漏标引号=N`。

### 人物词典（没有章节概念）

- 文本流到哪，词典维护到哪：`maintain_roster` 对**每段文本**增量维护，`speak` 的 role 一律写规范名，标注后把标签归一为规范名。
- **主词必须是全名/全称**：先见到简称、后见到全名时，全名会提升为主词，旧简称并进 aliases（同一人只留一条）。
- aliases 见到多少收多少、**不限数量**；**绝不收**指代/代词/整句/泛称/地名/组织/种族。
- **不写声线**：声线在 voicebank 阶段按人物书（`roles.json`）重新设计（年龄/性别 + 设计样本）。
- 引号是对话句子的一部分，**留在 `<角色>…</角色>` 标记内**，不做引号清理；`[笑]/[叹气]` 等 vocal event 标注不再产出（渲染器仍可识别）。

### 窗口 + 滚动摘要（`--batch N`）

前 N 章喂全文（`前情`，只用于判断说话人、不处理）；之后每章上下文 = `system`（提示词 + 人物词典）
+ `user`（`前情摘要` + 本章正文）+ `STEP_MARK`：

1. 章内跑完整 tool-loop（模型反复调用 `edit`）。
2. 章末 `compress()`：用「上一版摘要 + 本章出场角色（`parse_marks`）+ 本章正文」让 LLM 产出新的**滚动摘要**（≤400 字，写 `summary.txt`），只保留判断「谁在说话」需要的信息。
3. 下一章只带两样耐久记忆：**人物词典**与**前情摘要**；上一章正文和全部工具调用丢弃。
4. **无章末补漏**：编辑循环产出什么就是什么，只做机械收尾（别名归一 + `strip_quotes`）。

### 流水线可视化与覆盖层（`/workflow`）

产品 = **文本（段落流）**：没有章节对象，章节/台词/vocal event 都只是文本里的标记；所有阶段读写
同一段文本。结构（阶段、标记、参数、提示词）定义在 `scripts/workflow_store.py::PIPELINE`。

```
http://<host>:8899/workflow?book=<book>
```

- 页面逐阶段展示 入→出、脚本与命令；“标注”阶段可直接改 `batch / max_steps / think`
  和三个提示词（`LOCAL_SYSTEM / STEP_MARK / ROSTER_SYSTEM`），并可**一键保存覆盖并运行例章 ch003**（输入原文 / 输出 marked 两栏对照）。
- 保存写入 `outputs/<book>/workflow.json`，每次改动追加 `workflow_changelog.jsonl`（来源 web/agent 双向可见）；
  “清除覆盖”回到代码默认值。
- `mark_script.py` 启动时读取覆盖（覆盖优先于 CLI），启动日志打印 `[workflow] 覆盖: ...`。

### 实时查看

`mark_script` 边跑边写 `script/live.jsonl`，并生成自刷新页 `script/live.html`：

```
http://<host>:8899/outputs/<book>/script/live.html
```

页面按聊天气泡展示：人物挖掘、每次 `edit` 调用与结果、每章摘要。

**原始 LLM 输入/输出**（监督用）：`script/llm_raw.html`（请求的每条 message 可展开、原始 reasoning/content/tool_calls，1.5s 自刷新）与 `script/llm_raw.jsonl`（逐条 JSONL，含 roster 与标注两类调用）；每次运行开始会清空。

### 合并与间隔（后端）

- `audiobook/marks.py::parse_marks`：**相邻旁白自动合并成一段**（一次 TTS 生成）；相邻**同角色**台词合并。
- 同角色台词之间的短旁白**默认保留**（`AUDIOBOOK_MERGE_INTERRUPT_CHARS=0`，绝不丢旁白）；需要更连贯时可设正值（如 12/24），把 ≤N 字的短旁白吞并进台词——会丢字，慎用。
- 拼接间隔可调（`audiobook/tts.py`）：`AUDIOBOOK_GAP_SAME`（同角色，默认 0.25s）、`AUDIOBOOK_GAP_SPEAKER`（换角色 0.40s）、`AUDIOBOOK_GAP_KIND`（旁白↔台词 0.50s）。

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
