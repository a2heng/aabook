# 有声书"文本预处理/断句"实验日志

目标：在**纯 LLM**（不靠标点机械切）前提下，让 agent 产出「改声明 + 断句」的剧本，做到**问题最少**。
记录每次实验的方法、指标、结论，便于最后总结提高。指标来自 `outputs/<book>/script.json`（`scripts/build_script.py` 产物）。

## 指标约定

- **未闭合**：`tts_text` 不以 `。！？…` 结尾的段数（越少越好）。
- **碎片**：`target_duration_s < 2s` 的段数。
- **长段**：`>15s` 的段数（AuK 长生成会退化）。
- **anchor_failed / dialogue_edited / source_mismatch**：`flags` 计数。
- 测试章节：`outputs/book20_out/chapters/ch003.txt`（3518 字）。

---

## Exp-01 标签法抽取（plan-01，作废）

- 方法：Python 按引号边界切 span，LLM 只吐 `{i,kind,role}`；情绪靠后编辑。
- 问题：**大量第三人称叙述被判成 monologue**，整段旁白变成主角发言。
- 结论：弃用。抽取交给结构化 agent（Exp-02）。

## Exp-02 结构化抽取 + 审计 + 缺口回炉（plan-02）

- 方法：LLM 直接输出 `{kind,role,text}` 段；Python 回切对齐；audit agent 把误判的 monologue 降级；gap 缺口再分类。
- 问题：**归属短语（X说道）被并进对白**；纯叙述被标 dialogue；引号被改写。

## Exp-03 LLM 全文本改编 + 确定性闭合切分

- 方法：agent 输出**整段改编文本**（段落空行分隔），Python 按 `。！？…` 切整句。
- 结果：无碎片、无重复；但 agent 会**丢标点**，导致 run-on。
- 结论：可靠性取决于 agent 不动标点，不可控。

## Exp-04 指针断句（next 文本锚）+ 声明改写

- 方法：agent 每段给 `next=下一段原文开头 6~10 字`；Python 把锚匹配回原文定位切点（链表），kind/role 沿用抽取；再按"完整表达"合并碎片。
- 结果（第三章，77 段）：
  - ✅ 边界精确无吞并；垃圾段（模型吐的索引 `10`）已过滤；碎片 50→8；`<2s` 14→6。
  - ❌ **改编越权丢标点**（如 `…沟槽游走听不到卫队长的怒吼声，`），15 段 >15s（最长 20s）。
  - ❌ 对白/独白漏识别、归属误判（`赫蒂很严肃地说道，` 被标 dialogue；对白并进旁白）。
  - ❌ 8 段未闭合；`anchor_failed` 3；`dialogue_edited` 2。
- 结论：指针解决了"切点"，但没解决"只改声明"的约束——agent 仍在自由改写整段。

## Exp-05 edits-only（text 逐字 + speech 改写）——失败

- 方法：agent 每段输出 `text`（要求=原文逐字，用于定位/校验）、`next`（下一段锚）、`speech`（只改声明+气口的朗读文本）；`text` 与原文骨架不一致则标 `source_mismatch`。
- 结果（第三章，67 段）：
  - 段数 77→67，未闭合 8→7，`<2s` 6→7；但 **`>15s` 15→17**（最长 20s）更差。
  - `source_mismatch` 8：模型**没遵守 `text` 逐字**（如首段的 `text` 直接抄了后面章节/别的内容，raw 变成整窗重复）。
  - 仍在**跨段搬运/调序**（`s0001a..d` 的 text 来自别处），等于加戏。
  - `dialogue_edited` 4；run-on 仍在。
- 结论：**9B 模型无法可靠执行"逐字 text + next 锚 + 仅改 speech"这种强约束 schema**，会幻觉/调序。问题没解决，反而更不稳定。

### 对照（第三章）

| 实验 | 段数 | 未闭合 | <2s | >15s | 关键失败 |
| --- | --- | --- | --- | --- | --- |
| Exp-04 指针 | 77 | 8 | 6 | 15 | 丢标点/run-on、归属误判 |
| Exp-05 edits-only | 67 | 7 | 7 | 17 | source_mismatch、跨段搬运 |

## Exp-06 换模型：Spark-X2.5-4B（同一 edits-only schema）——成功

- 动机：Exp-05 的失败可能只是模型能力问题（Ornith-9B 不听强约束）。
- 服务器：`ckpts/llm/Spark-X2.5-4B-Q8_0.gguf`（`AUDIOBOOK_LLM_MODEL=spark-4b`、`AUDIOBOOK_LLM_PROFILE=spark-4b`，关闭 MTP spec）。
- 结果（第三章，67 段）：

| 指标 | Ornith-9B Exp-04(指针) | Ornith-9B Exp-05(edits-only) | **Spark-4B Exp-05** |
| --- | --- | --- | --- |
| 未闭合 | 8 | 7 | **1** |
| <2s 碎片 | 6 | 7 | **1** |
| >15s | 15 | 17 | 13 |
| source_mismatch | – | 8 | **2** |
| dialogue_edited | 2 | 4 | **1** |
| 对白识别数 | 10 | 9 | **33** |

- 结论：**主因是模型**。更小的 Spark-4B 反而**遵守逐字 text + next 锚 + 只改声明**，声明改写生效（`…声音从身后传来：`→`身后传来…声音：`），对白/独白识别从 10 升到 33，碎片/未闭合几乎清零。
- 残余问题：仍有 13 段 >15s（完整表达可以长，但逼近渲染 20s 上限）；个别归属仍可能被并进对白（抽取层）。
- HTML：`outputs/prep_spark/report.html`。

## Exp-07 TTS 单次长度极限（探测）+ 上限调整

- 方法：同一段 129 字，固定参考音（bwe），直接扫 `gen_seconds`（`duration_rate=1.0`，放开 20s 夹取），用 faster-whisper 统计**念出字数覆盖率**。
- 结果（129 字）：

| gen_seconds | 覆盖 | 结论 |
| --- | --- | --- |
| 15 | 86% | 太紧，漏字 |
| 20 | 95% | 略紧 |
| **25** | **100%** | 甜点 |
| 30 | ~100% | 仍可用 |
| 40 | 91% | 开始丢字 |
| 50 | 47% | 崩（并 OOM） |

- 结论：**单次生成上限约 25–30s**；太短也漏字。配合 `duration_rate=0.7`，一段估时可达 ~36s（约 150 字），比原 20s 上限多约 40%。
- 落地：
  - `RenderConfig.max_seconds`：20 → **28**（`renderer.py`）。
  - `duration.MAX_SEGMENT_SECONDS`：20 → **40**（估时；×0.7 ≤28s）。
  - 新增**旁白合并**：相邻同角色旁白在估时 ≤ `NARRATION_MERGE_SECONDS`（默认 36s）内合并成一段（`segment.py::_merge_narration`）。
  - `render_book.py` 增加 `--max-seconds`（默认 28）。

## Exp-06b Spark-4B 单章纯 LLM 耗时（不含模型加载）

- 第三章（3517 字）：提取 **25.1s**（8 调用）+ 预处理 **68.0s**（45 调用）= **93s**，产出 12.1 分钟音频，**RTF 0.128×**（预处理 93s ↔ 12min）。
- 外推 1595 章 ≈ 41 小时；优化方向：放大提取 chunk / 预处理窗口、多单元批量、投机解码（MTP/1.7B draft）。
- 复现：`scripts/bench_prep.py`（实验脚本已随代码清理删除）。

## 下一步候选（待定）

- **A. 保持 Spark-4B**，继续打磨（长段切分策略、抽取层归属修正）。
- **B. agents 标准化**（放后面）：schema 约束解码 / 窄任务 agent / 工具调用 + 校验。
- **C. 更大模型对照**（27B，暂缓）。

> 记录：Exp-04 HTML `outputs/prep_demo/report.html`；Exp-05 HTML `outputs/prep_demo5/report.html`；Exp-06 HTML `outputs/prep_spark/report.html`。

---

## 复现

```
AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=ornith-9b AUDIOBOOK_LLM_PROFILE=ornith-9b \
  .venv/bin/python scripts/build_script.py outputs/book20_out/chapters/ch003.txt \
  --out outputs/prep_demo --cast outputs/book20_out/cast.json
.venv/bin/python scripts/visualize_script.py --script outputs/prep_demo/script.json --out outputs/prep_demo/report.html
```

## Exp-07 单次输入/输出长度上限（Spark-X2.5-4B-Q8，ctx=16384）

探针 `scripts/probe_llm_len.py`（实验脚本已清理删除）：整段输入 → 输出 `{"segments":[{head,role,text}]}`（含整段正文），temperature=0.3。

| 输入字数 | prompt_tok | 输出字数 | 输出 tok | finish | 结果 |
| ---: | ---: | ---: | ---: | :--: | :-- |
| 400 | 382 | 689 | 884 | stop | ok, 7 段 |
| 800 | 650 | 1273 | 1245 | stop | ok, 10 段 |
| 1200 | 904 | 1853 | 1560 | stop | ok, 16 段 |
| 1600 | 1152 | **11608** | 8000 | **length** | **BAD（复读跑飞）** |
| 2200 | 1551 | 10881 | 8000 | **length** | **BAD** |

**结论：**
- 输出含整段正文时，输出量 ≈ 输入的 1.5–1.7×，且 **≥1600 字即触发复读** → 不能靠加大 chunk 解决。
- 要「一整章一次调用」，必须让**输出极小**：只回 `head`（前 6–10 字锚点）+ `role`，正文由代码按锚点切回来；长句加逗号只在需要时回一个 `speech` 短字段。
- 单次可靠上限（当前 schema）≈ **1200 字**；上下文 16384 够放整章（3518 字 ≈ 3–4k tok），瓶颈是**生成长度与复读**，不是 context。

## Exp-08 标签提示词变体对照（ch003，3518 字，head/tail+kind/role）

（脚本 `scripts/eval_labels.py` 已随 plan-02 清理移除）当时跑 v1–v4（含/不含示例、最小句切分、引号优先）：

| 变体 | units | narr | dlg | mono | dlg_no_quote | gap | roles |
| :-- | ---: | ---: | ---: | ---: | ---: | ---: | :-- |
| v1 | 61 | 30 | 29 | 2 | 0 | 0% | 赫蒂11/瑞贝卡6/精灵6 |
| v2 | 60 | 29 | 30 | 1 | 0 | 0% | 旁白12/瑞贝卡10 |
| v3 | 63 | 31 | 32 | 0 | 0 | 0% | 赫蒂12/瑞贝卡9 |
| v4(示例) | 61 | 29 | 32 | 0 | 0 | 0% | 瑞贝卡12/赫蒂12 |

- 四者都能 0 gap、且 `dlg_no_quote=0`（引号切分兜底，没有把旁白误当对白）。
- 差异主要在**角色归因**：v1/v3 偏向赫蒂，v2/v4 偏向瑞贝卡，说明**角色判定不稳**，需要 few-shot + 角色线索。
- 记录：`outputs/eval_labels.json`。

## Exp-09 扁平标签 + 整章一次调用（当前方案）

把任务压成两件事：**合理切断** + **每段标 `旁白`/`角色名`**。输出极小化以免 4B 复读：

- schema：`{"segments":[{"head":"段首6-10字","role":"旁白|角色名","speech":"可选，长句加逗号后的朗读文本"}]}`
- 代码按 `head` 定位切点；引号处再做结构切分（引号内→对白，引号外+归属短语→旁白）。
- 提示词内置 few-shot，覆盖易错点：归属短语算旁白、极短对白单独成段、连续对白各自成段、长句气口加逗号。
- few-shot 里刻意展示“`高文推开门，低声说道：`→旁白，不并入`你来了。`”“`拜伦！`极短仍成段”“长句只把加逗号文本放进 speech”。

**整章一次调用实测（ch003，3517 字，Spark-4B Q8）：**
- profile 参数（temp=1.0）：输出 6346 字、**116 段** head+role、无 `speech`、JSON 合法、约 **34s**，无复读。
- 直接 temp=0.3 / max_tokens=4096：输出 7017 字被截断、JSON 坏 → **必须用 profile 的 temp=1.0**，不要自己压低温。
- 116 段偏细（≈30 字/段，常按逗号断），但连续同 `kind`+`role` 的段会在 `_append_unit` 自动合并 → 下游 77 units，**过度切分无害**。

**结论：** 现在可以「一章 = 一次提取调用」（`AUDIOBOOK_EXTRACT_CHUNK_CHARS` 默认已放大），不链式。

**遗留：**
- 引号内容但说话人未知（`narration` 段里残留引号）→ 记为 `dialogue/旁白` 并打 `unresolved_role` 标记，交给 QA/RoleAgent。
- 输出是否稳定需要多章回归；`speech`（长句加逗号）这一轮模型几乎没用（116 段 0 个），需再调提示词或后置处理。

**复现：**
```
AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=spark-4b AUDIOBOOK_LLM_PROFILE=spark-4b \
AUDIOBOOK_EXTRACT_AUDIT=0 \
  .venv/bin/python scripts/build_script.py outputs/book20_out/chapters/ch003.txt \
  --out outputs/prep_flat --cast outputs/book20_out/cast.json
```

## Exp-10 编号标签法（numbered mode，当前方案）

**思路**：把「找边界」变成「给编号贴标签」。代码先把整章切成**引号感知的最小句**并编号，模型只回引号句的说话人；边界/合并/归属全部由代码决定。

- 输入示例：`1 高文推开门，低声说道：` / `2 “你来了。”` / `3 “好久不见。”`
- 输出示例：`{"speakers":{"2":"高文","3":"赫蒂"}}`（**只列引号句**，其余默认旁白 → 输出极小，避免截断）。
- 代码规则（结构性保证，不靠模型）：编号在引号外 → 一律旁白（**彻底消灭串味**）；引号内未列/列了未知角色 → 对白+`unresolved_role`；列成"旁白" → 当作非台词的引文。
- 实现：`audiobook/extract.py`（**唯一路径**；旧的 structured / head-tail 路径已在 plan-02 清理时移除）。

**ch003（3517 字，215 个最小句）实测：**
- 单次调用完成，`units=73`、`unresolved_role=10`（旧 flat 模式 31、structured 也不低）。
- 修好了旧模式的三个硬伤：**串味**（"她并没费什么功夫…"不再算赫蒂）、**相邻异人合并**（"愿先祖宽恕…"/"赫蒂姑妈，"、"这里…"/"这里便是…"、"姑妈？"/"这有可能是…"全部正确拆开）、**重复输出**。
- 残留：约 10 句短对白模型没写进 `speakers` → 落到 `unresolved_role`（可后置用就近归属短语补），见 `outputs/prep_num/`。

**结论**：编号法把模型任务压到「只判说话人」，输出短、结构完整、可校验，是目前最优。下一步：对未定引号句做「就近归属短语」回填。

**复现：**
```
AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=spark-4b AUDIOBOOK_LLM_PROFILE=spark-4b \
  .venv/bin/python scripts/build_script.py outputs/book20_out/chapters/ch003.txt \
  --out outputs/prep_num --cast outputs/book20_out/cast.json
# 报告固定写到 outputs/report.html（刷新浏览器即可）
```

## Exp-11 编号法提示词补丁：引号句不得遗漏

Exp-10 残留的 10 句"有人说话但未定角色"其实**原文都有引号**（如 `…疲惫，“我们至少能喘口气了。”`），是模型漏把它们写进 `speakers` 才落到 `unresolved_role`。

补丁：
- 提示词明确「**所有引号句都必须出现在输出里**」：是台词→写说话人（哪怕只有"拜伦！""姑妈？"）；不是台词（书名/术语，如"第一王朝"）→写 `"旁白"`；并加了 `他皱眉道：“你也在这儿？”` 的例子。
- 调用时把**待标的引号句编号清单**也塞给模型：`需要标注说话人的引号句编号（一个都不能漏）：2,3,5,...`。

结果（ch003）：`unresolved_role` **10 → 0**，`units=67`。抽查：
- `我们至少能喘口气了。`→瑞贝卡；`你好大的胆子！`→拜伦；`瑞贝卡！快离开那！`→赫蒂；`或许是地表的那些怪物……`→赫蒂；`“第一王朝”`→旁白。
- 个别漏网：`“卧槽谁砸我手！”`（棺中神秘人）被标成旁白并入叙述——身份未明，可后置就近归属再判。

产物：`outputs/prep_num2/`，报告 `outputs/report.html`。

## Exp-12 ASR 回检：AuK 克隆的不可靠性与时长规避（ch003）

方法：GPU `faster-whisper(small, float16)` 逐行 ASR，和该行 `tts_text` 做**拼音比对**（`scripts/asr_check.py`，容忍同音字）。指标：
- **coverage = ASR 音节数 / 期望音节数**（发现"吞句/截尾"），
- **score = 拼音序列相似度**（发现"念错/串词"）。
> ASR 仅用于**检查/观察**，不接入自动重做（见文末结论）。

### 收紧前（28s 上限，69 行）—— 按音节数分桶

| 行音节数 | 行数 | 平均 coverage | 最差 | 平均 score |
| --- | ---: | ---: | ---: | ---: |
| 0–40 | 46 | 1.00 | 0.92 | 0.96 |
| 40–80 | 8 | 1.00 | 0.99 | 0.99 |
| 80–120 | 5 | 1.00 | 1.00 | 0.99 |
| 120–150 | 5 | 0.97 | 0.93 | 0.98 |
| **≥150** | 5 | **0.78** | **0.46** | **0.83** |

特征相关性（收紧前）：逗号 ≥8 的行 mean_cov **0.81**、含 `——` mean_cov **0.91**、含 `……` 1.00、结尾 `：` 0.96、结尾 `，` 1.00。→ 长/多分句是主因，破折号次之。

最坏样例：`s0038`(163音节) cov 0.62、`s0041`(173) cov 0.46、用户点名的 `s0002`(149, 24s) cov 0.93（丢了"惨叫，更听不到那些恐怖怪物"半句）。

### 收紧后（20s 上限 + 每字×1.05 + 短句×1.10 + 正则清洗前置，77 行）

| 行音节数 | 行数 | 平均 coverage | 最差 | 平均 score |
| --- | ---: | ---: | ---: | ---: |
| 0–40 | 47 | 1.00 | 0.92 | 0.95 |
| 40–80 | 13 | 0.95 | 0.37 | 0.96 |
| 80–120 | 17 | 0.93 | 0.22 | 0.93 |

- ✅ `s0002` 变 **cov 1.00 / score 1.00**，长行灾难性截断消除（≥150 音节的行已被切碎）。
- ⚠️ 仍有零散失败（`s0010b` cov 0.22、`s0014b` 0.37、`s0005b` 0.56）：它们时长只有 8–16s，**语音占满整段但 ASR 只听到第一句**（能量包络显示声音到结尾，无静音尾巴）→ 属模型对某些文本段的生成退化，非时长问题。
- 🐛 顺带发现一行 `tts_text="。"`（纯标点，被念成"呃"）→ 已在 `segment_tts_text` 过滤纯标点片段。

### 结论

1. **AuK 克隆不可靠**：长段/难段会吞句、串词或生成退化；**时长控制是目前最有效的工程规避**（收紧上限 + 每字 +5% + 短句 +10% + 清洗前置），大幅减少但不根治。
2. **ASR 只做体检**：用于量化 coverage/score、定位问题行；不接入自动重做（重做需换 seed/改文本/调时长再复查，过于繁琐，且收益不确定，暂缓）。
3. 若要上线，建议：跑完 **ASR 体检 → 人工挑 ≤ 少量行** 重录；不追求全自动闭环。

**复现：**
```
# GPU ASR 体检（faster-whisper 需 cu12 运行库，用 LD_LIBRARY_PATH 指到 nvidia-cublas-cu12/cudnn-cu12）
LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cublas/lib:.venv/lib/python3.12/site-packages/nvidia/cudnn/lib \
  .venv/bin/python scripts/asr_check.py --script outputs/ep003/script.json \
  --rows outputs/ep003/audio/rows --out outputs/ep003/asr_check.json
```
