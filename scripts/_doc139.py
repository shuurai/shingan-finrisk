# -*- coding: utf-8 -*-
"""Record the real-data zero-shot result (69.5h run, artifact 20260926T075133Z)."""
from pathlib import Path

ROOT = Path(r"D:\workspace\shingan-finrisk")
LF = chr(10)


def swap(path, old, new, label):
    p = ROOT / path
    t = p.read_text(encoding="utf-8")
    if label in t or (new.strip() and new.strip() in t):
        print(f"SKIP  {label} (already present)")
        return
    n = t.count(old)
    assert n == 1, f"{label}: anchor count {n} (expected 1)"
    t = t.replace(old, new)
    p.write_bytes(t.replace(chr(13) + chr(10), LF).encode("utf-8"))
    print(f"OK    {label}")


def insert_before(path, anchor, block, label):
    swap(path, anchor, block + anchor, label)


# ---------------------------------------------------------------- docs/09
D09 = "docs/09-lora-evaluation.md"

section_13_9_1 = """#### 13.9.1 落地：69.5 小时后的真实零样本数字（2026-09-26）

第三次尝试（`--batch-size 1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）于
2026-09-26 07:51 落盘，墙钟 **69 h 31 m**，产物 `artifacts/lora-eval/20260926T075133Z`。

**真实面板 test（`tail_risk`）上的完整对比：**

| path | role | AUC | KS | KS dir | PR-AUC | 正样本 | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| structured | baseline | 0.6294 | 0.4407 | negatives_higher | 0.0160 | 5 | 492 |
| structured_matched | baseline | 0.7497 | 0.5072 | positives_higher | 0.0248 | 5 | 492 |
| text_baseline | baseline | 0.3306 | 0.5671 | negatives_higher | 0.0112 | 5 | 492 |
| fused | baseline | 0.6940 | 0.5585 | positives_higher | 0.0192 | 5 | 492 |
| text_only_zero_shot | **arm** | **0.7542** | **0.5451** | **positives_higher** | **0.0960** | 5 | 479 |

配对差值（零样本 − `structured_matched`；正确基准是它，因为零样本 prompt 携带同一组
12 列 `<STRUCTURED_SIGNALS>`，见产物 caveat）：

- AUC **+0.0034** [−0.3491, +0.4367]，跨零
- PR-AUC **+0.0702** [−0.0192, +0.2354]，跨零

492 行里 13 行不参与配对——零样本臂无分数。原因**不是 schema**：解析守卫记的是
"model answered label `default_risk` but `tail_risk` was requested"，模型答错了标签名。
13 行里没有正样本，丢弃不改变正类样本量。

**能写的三条：**

1. **契约修复外推成功。** 合成面板上修好 `SYSTEM_PROMPT` 后零样本 4/4 → 真实面板
   **479/492 解析（96.2%）**、方向 `positives_higher`——底座模型把真实 tail-risk
   事件排在了无事件之上。文本轨第一次有真实数据上的零样本基线；此前的相关表述
   （含本文件 12.6 的三个选项）全部是推断，从这一行起被测量替代。
2. **点估计上零样本不落后于同信息集的 `structured_matched`**（AUC 0.7542 vs 0.7497；
   PR-AUC 0.0960 vs 0.0248，约 3.9 倍）。**这不构成性能结论**：两个区间都跨零，
   test 只有 5 个正样本，PR-AUC 的区间下界 −0.019 离负值只差毫厘。可写"值得跟踪
   的信号"，不可写"文本轨赢了"。
3. **条件性与保真度**：指标以"模型产出可用输出"为条件（479/492，与解析失败丢行
   同一偏差类）；且本次运行启动于 Step 9 之前，产物是**旧 schema**——没有
   `citations` 段，它的引用保真度没有被审计。下一次重跑的产物才会带上引用审计。

**成本记录**（这条路径此前从未被量过）：

- 墙钟 **69.5 h**：bs=1、4096 token、4-bit。用 `py-spy dump --locals` 两次读运行中
  进程的循环变量（407/492 → 462/492 → 完成），实测平均 **479 s/行 ≈ 8.4 tok/s**。
- 根因：4-bit bitsandbytes 在 Blackwell（5090）上没有调优的反量化 kernel，bs=1 时
  每生成一个 token 都要反量化全部权重。显存时钟满频、功耗仅 135–146 W（带宽受限）、
  温度 40–41°C——"utilization 100% 但低温"不是挂起，是解码的正常签名。
- 运行中途（24 h 处）补了可观测性（提交 `7b59e9a`）：进度行带每 batch 耗时、
  `eval lora` 默认开启。后续评测应提供 **bf16 加载选项**（14B bf16 解码预期
  40–60 tok/s，快 5 倍以上，显存足够）；切换前须在 smoke 上验证两种加载的数字一致。

"""

swap(
    D09,
    "### 13.10 这一步没验证什么",
    section_13_9_1 + "### 13.10 这一步没验证什么",
    "docs/09 §13.9.1 appended",
)

swap(
    D09,
    "它的两次尝试见 13.9。两臂的配对差值至今只在合成面板上成立。",
    "它的三次尝试见 13.9——第三次于 2026-09-26 落地，真实零样本数字见 13.9.1。两臂的"
    "配对差值至今只在合成面板上成立。",
    "docs/09 §13.10 bullet updated",
)

swap(
    D09,
    "真实零样本行（`--max-new-tokens 4096` 版）仍在跑；其结果将按 §13.9 的口径补录。",
    "真实零样本行已于 2026-09-26 落地（69.5 h），结果见 §13.9.1；产物为旧 schema（无引用审计）。",
    "docs/09 §14.5 zero-shot bullet updated",
)

swap(
    D09,
    "真实零样本行占用，等它落地后执行。",
    "GPU 已空闲（真实零样本行 2026-09-26 落地，见 §13.9.1），随时可执行。",
    "docs/09 §14.5 placebo bullet updated",
)

# ---------------------------------------------------------------- docs/05
swap(
    "docs/05-evaluation.md",
    "这张表直接对应 `shingan eval compare --runs structured text fused`。",
    "**真实数据上的零样本行（2026-09-26 落地）**：契约修复外推成功——真实 test 块"
    "（492 行 / 5 正样本）**479/492 解析**（13 行因答错标签名被守卫拒绝，正样本零丢弃），"
    "**AUC 0.7542 / KS 0.5451 / PR-AUC 0.0960 / 方向 `positives_higher`**。注意该行 prompt"
    " 携带 12 列结构化信号，正确的对照是 `structured_matched`（同样的 12 列）：配对 AUC"
    " **+0.0034** [−0.3491, +0.4367]、PR-AUC **+0.0702** [−0.0192, +0.2354]，区间均跨零。"
    "可写：零样本基线已实测、点估计不落后、值得跟踪；不可写：文本轨赢了——5 个正样本上"
    "没有任何数字过得了那一关。产物 `artifacts/lora-eval/20260926T075133Z`（69.5 h 墙钟，"
    "bs=1 + 4-bit ≈ 8.4 tok/s；产物早于引用审计，引用未审计）。详见"
    " [09](09-lora-evaluation.md) §13.9.1。" + LF + LF
    + "这张表直接对应 `shingan eval compare --runs structured text fused`。",
    "docs/05 real zero-shot paragraph",
)

# ---------------------------------------------------------------- README
swap(
    "README.md",
    "| Zero-shot arm `text_only_zero_shot` (`--mode both`) | Produced · **not citable**",
    "| Zero-shot arm `text_only_zero_shot` (`--mode both`) | Produced · **measured on both panels**",
    "README zero-shot row header",
)

swap(
    "README.md",
    "Not citable because the panel is synthetic: the text signal in it is planted by the generator, so this is a check on the wiring and zero evidence about markets |",
    "Not citable on the synthetic panel because the text signal is planted by the generator. **Real panel (492 rows / 5 positives): 479/492 parse, AUC 0.7542 / KS 0.5451 / PR-AUC 0.0960, direction `positives_higher`**; paired vs `structured_matched` (the same twelve signals the prompt carries): AUC +0.0034 [−0.3491, +0.4367], PR-AUC +0.0702 [−0.0192, +0.2354] — both CIs cross zero. A tracked signal, not a performance claim; the artifact predates the citation audit, so its `source_ref`s are unaudited |",
    "README zero-shot row real result",
)

readme_bullet = """- **The real-data zero-shot baseline exists, and the point estimates favour the base model.** Run on the real test block (492 rows, 5 positives; 69.5 h at batch 1 on a 4-bit 14B), the base model under the fixed contract parses 479/492 — the 13 failures all answered the wrong label name, and no positive was dropped — and posts AUC 0.7542 / KS 0.5451 / PR-AUC 0.0960 with the ranking pointed `positives_higher`. Against `structured_matched` (the same twelve structured signals the prompt carries) the paired differences are AUC +0.0034 [−0.3491, +0.4367] and PR-AUC +0.0702 [−0.0192, +0.2354] — both cross zero. Readable: the instruction-contract fix extrapolates to real filings, and the text track's zero-shot baseline is a measurement now, not an inference. Not readable: "text beats structured" — with 5 positives nothing here clears that bar. The artifact predates the citation audit, so its citations are unaudited. Full record: [09 LoRA evaluation](docs/09-lora-evaluation.md), section 13.9.1.
"""

swap(
    "README.md",
    "- **SEC filing bodies are ingested and the evaluation has been re-run on the new panel.**",
    readme_bullet + "- **SEC filing bodies are ingested and the evaluation has been re-run on the new panel.**",
    "README honesty bullet inserted",
)

# ---------------------------------------------------------------- docs/index.md
swap(
    "docs/index.md",
    "真实数据上的零样本行见 [09](09-lora-evaluation.md) 第 13 节 |",
    "真实面板（492 行 / 5 正样本）**479/492 解析，AUC 0.7542 / KS 0.5451 / PR-AUC 0.0960、方向 `positives_higher`**；对 `structured_matched` 的配对差 AUC +0.0034、PR-AUC +0.0702，区间均跨零——值得跟踪的信号，不是性能结论；产物早于引用审计。见 [09](09-lora-evaluation.md) §13.9.1 |",
    "index zero-shot row updated",
)

swap(
    "docs/index.md",
    "零样本臂在修掉指令契约缺口后**已经能测且不是常量**（两臂都 64/64 解析），但它在合成面板上，所以读不出能力；",
    "零样本臂在修掉指令契约缺口后**已经能测且不是常量**，且真实面板上有了第一个实测基线（AUC 0.7542、方向正确，配对区间跨零，见 [09](09-lora-evaluation.md) §13.9.1）；",
    "index honesty paragraph updated",
)

print("all done")
