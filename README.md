# Speculative Decoding Lab

> 从零实现投机解码（Speculative Decoding）：**正确性可证明、加速可解释、结果可复现**。
> 不依赖 `transformers` 的 `assisted_generation`，draft/verify 循环、KV Cache 截断、拒绝采样全部手写。

**English overview**: A from-scratch implementation of speculative decoding with exact
draft/verify rejection sampling, explicit KV-cache rewinding, distributional
correctness proofs (exact joint-distribution chi-square test on an enumerable toy
model, KL/TV tests on a real model pair), and a measured-vs-analytic study of the
speed-up optimum `γ*`.

---

## 1. 这个项目解决什么问题

自回归解码每生成 1 个 token 就要把整个大模型读取一遍显存，是**显存带宽受限**而非算力受限。
投机解码的思路是：让一个小模型（draft）连续猜 `γ` 个 token，再让大模型（target）**一次前向**
并行验证这 `γ` 个 token，一次前向最多可以确认 `γ+1` 个 token —— 用同样的带宽，换更多的 token。

难点有两个，本项目都做了完整处理：

1. **数学正确性**：验证环节必须是**拒绝采样**（accept with `min(1, p/q)`，拒绝时从归一化的
   `(p - q)₊` 重采样），否则输出分布会被 draft 模型带偏。
2. **工程正确性**：拒绝发生时，target 的 KV Cache 必须**回滚**，否则后续 token 全都基于错误的上下文。

---

## 2. 核心设计

### 2.1 每轮只做一次 target 前向

朴素实现会在"接受全部 token 后采样 bonus token"和"拒绝后重采样"两个分支上额外补一次
前向去同步 Cache，这会吃掉一半加速。本项目用一个统一的不变量消除它：

> **缓存不变量**：每轮结束后，`past` 只保存位置 `0 .. L-2` 的 KV，最后一个已确认的 token
> `sequence[L-1]` 保持"未喂入（pending）"状态。

于是每一轮 target 只做一次前向，输入是 `[pending] + draft_tokens`（共 `γ+1` 个位置）：

| 前向输出 | 含义 | 用途 |
| --- | --- | --- |
| `ℓ[0]` | 第 0 个 draft token 的分布 `p₀` | 验证 draft token 0 |
| `ℓ[1..γ-1]` | 第 1..γ-1 个 draft token 的分布 | 验证 draft token 1..γ-1 |
| `ℓ[γ]` | bonus 位置的分布 `p_γ` | 全接受时采样 bonus token |

拒绝发生在第 `i` 位时，把缓存截断到 `L+i`（`DynamicCache.crop`），
被拒绝的 token 及其后续全部从缓存中消失，新重采样的 token 成为下一个 pending token。

`tests/test_speculative.py::test_one_target_forward_per_round` 直接把这条性质钉死为断言：

```
target.forward_calls == rounds + 1        # +1 是 prompt prefill
```

### 2.2 算法流程

```
输入: prompt, γ, draft 模型 Q, target 模型 P
循环直到生成 max_new_tokens:
  1. draft 用自身缓存自回归生成 γ 个候选 x₀..x_{γ-1}，记录 q₀..q_{γ-1}
  2. target 一次前向 [pending] + [x₀..x_{γ-1}] → p₀..p_γ
  3. for i in 0..γ-1:
       以 min(1, pᵢ(xᵢ)/qᵢ(xᵢ)) 的概率接受 xᵢ
       若拒绝: xᵢ' ~ normalize((pᵢ - qᵢ)₊)，接受它并跳出
  4. 若全部接受: bonus ~ p_γ（由 target 自己产生）
  5. 截断 target/draft 缓存，恢复不变量
```

贪心模式（`temperature <= 0`）不需要单独分支：`p` 退化为 one-hot，接受概率自然变成
"argmax 是否一致"，残差分布退化为 argmax 本身。

---

## 3. 正确性验证（三层）

| 检验 | 方法 | 强度 |
| --- | --- | --- |
| `greedy_equivalence` | 贪心模式下与自回归解码**逐 token 完全一致** | 确定性，必须完全相同 |
| `first_token_test` | 首个 token 的经验分布 vs target 的**解析分布**，报告 KL / TV / 卡方，并以**同 N 下的直接采样噪声下限**作为对照 | 中等 |
| `sequence_distribution_test` | 玩具小模型（词表 8）**枚举全部 `8² = 64` 种 2-token 结果**，把投机解码与自回归解码的经验联合分布同时和**精确联合分布**做卡方检验 | 最强，直接证明拒绝采样数学正确 |

第三项是关键：任何 `(p - q)₊` 重采样写错、或 Cache 截断漏掉一个位置，都会立刻表现为
卡方检验被拒（`p < 0.01`）。

---

## 4. 加速比为什么有一个最优 γ

设单 token 接受率为 `α`，draft 单步成本是 target 单步成本的 `c` 倍，则：

```
每轮期望提交 token 数  E[τ] = (1 - α^(γ+1)) / (1 - α)      ← 随 γ 饱和到 1/(1-α)
每轮相对成本           cost = 1 + γ · c                     ← 随 γ 线性增长
预测加速比             S(γ) = E[τ] / cost
```

分子饱和、分母线性增长 ⇒ **加速比在有限 γ 处取到最大值 γ\***，继续加大 γ 只会让 draft 白算。
`specdec/analysis.py` 会输出：

- `γ*` 的解析解，以及边际收益 `ΔS/Δγ` 何时降到 0；
- 实测加速比与解析预测的**相关系数**（衡量解析模型是否把 γ 排序排对了）；
- 实测最优 γ 与预测最优 γ 是否一致。

---

## 5. 快速开始

```bash
# 1. 环境（Python >= 3.8）
pip install -r requirements.txt

# 2. 单元测试（玩具模型，无需下载权重，约 25 秒）
python -m pytest -q

# 3. 正确性证明（玩具模型，枚举全部结果）
python scripts/run_correctness.py --mode toy --gamma 4 --num-samples 2000

# 4. 加速比基准 + γ 扫描（需要下载 draft/target 权重）
python scripts/run_gamma_sweep.py \
    --draft-model Qwen/Qwen2.5-0.5B \
    --target-model Qwen/Qwen2.5-1.5B \
    --gamma 1,2,3,4,5,6,8,10,12 \
    --max-new-tokens 64 --repeats 3
# GPU 上建议加 --use-cuda-graph：把 draft 单步前向捕获为 CUDA Graph，
# 消除逐 kernel launch 开销（成本比 c 从 0.86 降到 0.118，见第 6 节）

# 跨 tokenizer 的 draft 自动走文本级路径（greedy-only，实测无加速，见第 6 节）
python scripts/run_benchmark.py \
    --draft-model HuggingFaceTB/SmolLM2-135M \
    --target-model Qwen/Qwen2.5-1.5B \
    --device cuda --max-new-tokens 64 --repeats 3

# 5. 出图（PNG + PDF）
python scripts/make_plots.py
```

国内下载 HuggingFace 权重建议先设置镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com     # Windows PowerShell: $env:HF_ENDPOINT="..."
```

---

## 6. 结果

实验在 Qwen2.5-0.5B（draft）→ Qwen2.5-1.5B（target）组合上进行，每个 γ 扫描 9 档
（1/2/3/4/5/6/8/10/12），3 条提示词、`max_new_tokens=64`、`repeats=3`、种子 `20260920`，
分 CPU（float32）与 GPU（float16 / RTX 4060）两条路径，结果见 `results/cpu/` 与 `results/gpu/`。
GPU 额外实现了 **CUDA Graph 优化**（`specdec/graph_draft.py`：把 draft 的固定形状单步前向
捕获为一张图、每次 replay 一次提交，消除逐 kernel 的 Python/调度/launch 开销），
用 `--use-cuda-graph` 开启。

| 设备/模式 | draft 成本 c | 最优 γ | 最优加速比 | 平均接受率 ᾱ | 实测 vs 解析 corr | 基线 ms/token |
|-----------|-------------|--------|-----------|--------------|------------------|---------------|
| CPU (eager)    | 0.40  | 3  | 1.04×  | 0.50 | 0.945 | 329.6 |
| GPU (eager)    | 0.86  | 2  | 0.69×  | 0.50 | 0.914 | 47.8  |
| GPU (CUDA Graph, 0.5B→1.5B) | **0.118** | 5 | **1.81×** | 0.36 | 0.157 | 75.1 |
| GPU (CUDA Graph, 1.5B→3B)   | 0.248 | 6 | **3.97×** | 0.60 | −0.277 | 333.7 |
| GPU (text-level, SmolLM2-135M) | 1.02 | 1 | 0.54× | 0.73 | 0.952 | 50.1 |

四点结论，均从数据中直接读出，且如实说明边界：

0. **更大的同 tokenizer draft 是决定性的**。从 0.5B→1.5B 换成 1.5B→3B（基线 ms/token
   从 75.1 升到 333.7），draft 成本 c 只从 0.118 升到 0.248（CUDA Graph 把 1.5B draft 的
   单步固定开销也压得很低），而换来接受率从 0.36 升到 0.60——最优加速比从 **1.81×** 跳到
   **3.97×**（γ=6），3 条提示词平均 **4.14×**（3.87 / 3.13 / 5.43）。这直接验证了"提高 draft
   质量 > 减小参数"：E[τ] 随接受率近线性增长，而 `1+γc` 只从 1.4 升到 2.2，净效果是加速翻倍。
   全 γ 档（1–12）均快于基线（2.26× 起步），γ=1 时接受率高达 0.97。
1. **CUDA Graph 是决定性优化**。GPU 上 draft 单 token 前向从 52.3 ms 降到 6.6 ms（7.9×），
   成本比 c 从 0.86 降到 0.118，投机解码从 0.69× 提升到 **1.81×**（γ=5），3 条提示词
   平均 **1.56×**（1.74 / 1.37 / 1.57）。根因是 batch=1 下 draft 前向的 FLOPs 优势被
   Python 调度与逐 kernel launch 开销吞没（`1+γc` 中 c≈1 时无利可图），CUDA Graph
   把整段固定形状的前向合并为一次提交，把固定开销几乎清零。
2. **接受率仍是天花板**。α 从 γ=1 的 0.71 单调衰减到 γ=12 的 0.26，
   `E[τ]≈1+ᾱ·γ` 使每轮提交 token 数在 γ≥6 后饱和（~3.9），继续拉长草稿只增加 draft
   成本，因此最优 γ 停留在一个中等值，γ≥8 后加速比回落。
3. **解析模型在 graph 模式下系统性低估加速比**（0.5B→1.5B corr 0.16；1.5B→3B 更甚，
   corr=−0.28）。`E[τ]=(1−α^(γ+1))/(1−α)` 假设每个位置接受概率恒等于全局平均 ᾱ，但真实
   接受率在轮内随位置快速衰减，用全局平均会严重低估前几个 draft token 的高接受概率，而
   正是前几个 token 贡献了绝大部分 E[τ]，于是解析曲线随 γ 塌陷（1.5B→3B 下预测峰值仅
   2.27 vs 实测 3.97，形状都不匹配）。eager 模式下 corr 0.91–0.95 是因为 c≈0.86 时
   `1+γc` 主导、E[τ] 误差被掩盖；c 降到 0.118–0.25 后该偏差成为主要误差来源。
   这也说明：优化方向不是盲目加大 γ，而是提高前几个草稿 token 的质量（更好的 draft）。

**额外的负面结果：跨 tokenizer 的"更小 draft"不可行**。为验证"换更小的 draft 模型"，
实现了文本级投机解码（`specdec/text_speculative.py`，draft 用自己的 tokenizer 提议、
解码成文本增量后由 target 重编码验证，greedy-only），并尝试 SmolLM2-135M → Qwen2.5-1.5B
（词表 49152 vs 151936，自动走文本级路径）。结果在**所有 γ 下都慢于基线**（最优 γ=1，
0.54×；γ=12 仅 0.23×），corr=0.952。原因有二：

- **c≈1.02 没有任何预算空间**：batch=1 下 135M draft 的单 token 前向耗时与 1.5B target
  几乎相同（都是显存带宽受限，FLOPs 优势被掩盖），`1+γc` 的代价在 γ=2 就翻倍；
- **文本级对齐开销巨大**：每轮除 γ 次提议外，提交后必须把已确认文本重新 encode 并用
  LCP 截断重建 draft KV cache，实测 draft 前向调用数是 target 的 9 倍（204–294 vs 21–39）；
  且跨词表的 token 粒度不对齐使按 target token 计的接受率天然偏低（0.11–0.38）。

这进一步印证第 3 点的结论：加速来自"压低 c 的同 tokenizer 更强 draft"（CUDA Graph +
更大/更准的草稿模型），而非单纯减小参数量。文本级路径作为 greedy 正确性验证设施保留，
保证跨 tokenizer 组合不会给出错误输出。

**批量（bundle size）实验：c(B) 随 batch 下降，但不足以跨过 1×**。为验证"把 B 份相同
序列组成一个 bundle 能提高 kernel 利用率、压低 draft 成本比 c"，新增同序列批量路径
（`specdec/batch_speculative.py`：B 份相同序列在 batch 维上前向、进度严格同步，接受逻辑
退化为单样本规则；入口 `scripts/run_batch_experiment.py`）。greedy 等价 probe 通过：
任取一份逐 token 等于 `baseline_decode` 且与 B 无关（B=1/2/4 完全相同，接受率恒 0.395、
rounds 恒 19）。结果（Qwen2.5-0.5B→1.5B，GPU eager，γ=6，`results/gpu/batch_experiment_*`）：

| batch B | draft 成本 c(B) | draft ms/step | target ms/step | spec ms/slot | vs 同 B baseline 加速比 | 峰值显存 |
|--------|---------------|--------------|----------------|-------------|----------------------|---------|
| 1 | 1.079 | 51.1 | 47.4 | 98.3 | 0.48× | 3921 MB |
| 2 | 0.885 | 40.5 | 45.7 | 100.8 | 0.46× | 3926 MB |
| 4 | 0.847 | 39.8 | 46.9 | 101.4 | 0.46× | 3937 MB |

**机制被证实、结论仍是否定性的**。draft 单步成本随 B 从 51.1 降到 39.8 ms（−22%），
c 从 1.079 降到 0.847（−21.5%）——这证实了"bundle→c 下降"这一子机制真实存在：
batch 把 draft 欠利用的 kernel 填满，缩小了它与 target 之间的利用率差距。但两个下界同时成立：
(i) 该路径走 eager（批量 CUDA Graph 引擎未实现，见 YAGNI 免责），c 即便降到 0.85 也远高于
让提速恢复 >1 的水平——对比而言，CUDA Graph 把 c 从 0.86 压到 0.118（约 86%），而 batch 只压
约 20%，相差一个量级；(ii) target 的 batch kernel 同样变快（同 B baseline 吞吐 21.0→21.5
tokens/s/slot），故投机相对基线的优势并未随 B 放大，per-slot 加速比持平约 0.46×、绝对 spec
延迟反而微升（98.3→101.4 ms）。结论：**batch 化确实能压低 c，但缺少 CUDA Graph + 更强 draft
两个决定性杠杆时，仅靠 bundle 无法把提速拉到 1× 以上**——与此前"c≈1 区间无利可图"的结论一致。
峰值显存从 B=1 的 3921 MB 到 B=4 仅 +16 MB，证明 KV 的 batch 维开销很小，8 GB 显存下 B≤4 安全。

正确性方面，玩具模型三层检验（greedy 等价、首 token 精确分布、联合分布枚举）全部 PASS；
真实模型在 GPU 上 3 条提示词的 greedy 等价 + 首 token 分布检验（KL/TV/χ²，N=2000）全部 PASS，
其中 `--use-cuda-graph` 模式的 greedy 等价同样 PASS（CUDA Graph 路径与 DynamicCache 路径
逐 token logits 完全一致）。图见 `results/cpu/figures/` 与 `results/gpu/figures/`（fig1–5，PNG + PDF）。

---

## 7. 目录结构

```
speculative-decoding-lab/
├── specdec/
│   ├── sampler.py        # logits 后处理、采样、(p-q)₊ 残差分布、KL/TV
│   ├── models.py         # 带手动 KV Cache 控制的模型封装 + 玩具模型对
│   ├── decoding.py       # 自回归基线解码（所有加速比的分母）
│   ├── speculative.py    # 投机解码主循环 + 解析加速模型
│   ├── batch_speculative.py # 同序列批量投机解码（B 份同 prompt，bundle 尺寸对 c 的影响）
│   ├── correctness.py    # 三层正确性检验
│   ├── graph_draft.py    # CUDA Graph 加速的 draft 引擎（StaticCache + 图捕获）
│   ├── text_speculative.py  # 跨 tokenizer 文本级投机解码（greedy-only，用于不同词表的 draft）
│   ├── benchmark.py      # 计时/显存/接受率测量与 γ 扫描
│   ├── analysis.py       # γ* 解析最优解与边际收益分析
│   └── utils.py          # 日志、随机种子、JSON 序列化、CJK 字体
├── scripts/              # CLI 入口（正确性 / 基准 / 扫描 / 批量实验 / 出图）
├── tests/                # pytest：采样器单测 + 解码循环不变量
└── results/              # JSON 报告与 PNG/PDF 图
```

---

## 8. 复现约定

- 所有随机性都来自显式 `--seed`（默认 `20260920`），采样统一在 CPU 上用同一 generator 完成，
  保证 CPU / CUDA 两条路径可比较。
- 每组配置先 warmup 再重复 `--repeats` 次，取均值与标准差；GPU 端计时前 `torch.cuda.synchronize()`。
- 显存取 `torch.cuda.max_memory_allocated` 的峰值。
- 退出码：`0` 成功，`1` 正确性检验未通过，`2` 参数非法或运行失败。

---

## 9. 已知局限

- 已实现**同序列批量**路径（`batch_speculative.py`，B 份相同 prompt、锁步前向）用于验证 bundle
  尺寸对 c 的影响；**异构 batch（不同提示词）仍需按样本独立回滚，是后续 Phase 2 工作**。批量
  路径未捕获 CUDA Graph 引擎，且仅支持 greedy（采样模式的批量拒绝采样未验证）。
- 跨 tokenizer 组合通过文本级路径（`text_speculative.py`）保证 greedy 正确性，但实测无加速
  （c≈1.02 + 每轮 KV 重建开销），采样模式（非 greedy）在跨词表时无定义，仅支持 greedy。
- 未实现 tree attention / Medusa 式多头草稿，`γ*` 分析仅针对链式 draft。
- toy 模型的精确枚举限于 `词表^seq_len <= 4096`，更长的联合分布无法枚举。

## License

MIT © 2026 zzj