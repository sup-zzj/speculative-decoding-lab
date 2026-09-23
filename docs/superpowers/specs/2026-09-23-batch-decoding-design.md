# 批量解码最小科学验证 — 设计文档（2026-09-23）

## 目标

验证核心假说：**batch（bundle 尺寸）增大时，draft 的成本比 c 是否因 kernel 利用率提高而下降，并带来每 slot 吞吐加速**。

范围限定（最小验证）：只测 `c(B)` 与 `加速比(B)`，输出 1 张图 + README 一小节。不扩展全部基准管线。

## 硬件约束（已实测）

- GPU：NVIDIA RTX 4060 Laptop，**8 GB**（8188 MiB），当前空闲约 4.25 GB。
- 模型组合限 **0.5B → 1.5B**（Qwen2.5，fp16 权重合计约 4 GB）。
- batch ∈ {1, 2, 4}，遇 OOM 时降为 {1, 2}（可参数化）。

## 关键决策：同序列 batch（Phase 1）

把同一条 prompt 复制 B 份构成 batch。理由：

- 投机解码 batch 的难点在于"批内各样本接受数不同 → 进度错位 → 需 per-sample 回滚 + 动态 padding"，复杂度高。
- 同 prompt 复制后所有样本进度严格同步，接受逻辑退化为统一逻辑，仅把前向维度扩成 B。
- "bundle 尺寸对 c 的影响"这一核心假说被完整保留（draft/target 均以真 batch 前向）。

异构图（不同 prompt）的 per-sample 回滚留待后续 Phase 2，本次不做。

## 实现改动

### 1. `specdec/models.py` 新增 batch 接口
- `prefill_batch(batch: list[list[int]]) -> (PastType, logits[B, ·, vocab])`
- `extend_batch(token_ids_2d: LongTensor[B, L], past) -> (PastType, logits[B, L, vocab])`
- KV cache 增加 batch 维；变长输入左填充，复用 attention_mask 掩盖 padding。

### 2. 新建 `specdec/batch_speculative.py`
- `speculative_decode_batch(prompt_ids, batch_size, gamma, ...)`：
  - 同 prompt 复制 B 份；
  - 逐轮：draft 提议 `[B, γ]` → target `extend_batch([B, γ+1])` → 逐位 argmax 接受（统一）→ 提交；
  - 统计 walltime、forward_calls、tokens_per_second（按 B 份输出 token 总量计吞吐）。

### 3. 新建 `scripts/run_batch_experiment.py`
- batch ∈ {1, 2, 4} 复选；
- 输出 `c(B)` 与 `加速比(B)`（相对 batch=1 baseline 总耗时）为 JSON + PNG/PDF 图（左轴 c vs B，右轴加速比 vs B）。

### 4. 成本比 batch 版
- `measure_draft_cost_ratio` 复用 `extend_batch` 单 token 前向：draft 耗时 / target 耗时，得 `c(B)`。

## 验证

greedy 等价：batch 结果任取一份，逐 token 等于 `baseline_decode`（同 prompt），且与 B 无关。

## 输出

- `results/gpu/figures/batch_c_speedup_vs_batch.png/pdf`
- README 新增小节（如实标注"同序列 batch，异构边界未验证"）
- 项目记忆同步（用户要求"及时归纳总结同步"）

## 明确不做（YAGNI）

- 异构 batch 的 per-sample 回滚（Phase 2）
- 批量 γ 扫描 / 批量 CUDA Graph 引擎改造
- 超出 8 GB 的任何模型组合