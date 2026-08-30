# llama.cpp 性能工程项目计划(数据校准版)

> **版本说明**:本版基于 2026-08-30 已完成的实验数据重写。原计划假设"从零开始、12 周线性学习";本版以你**已经拿到的真实 M2 数据**为起点,把剩余时间压缩为 **3 个里程碑 + 收尾**。
>
> 已有实验数据见 [docs/analysis.md](./analysis.md)。

---

# 0. 你现在已经拥有的(不是从零开始)

不要重跑已经做过的事。以下资产已经存在:

```text
[✓] 固定 llama.cpp commit(branch: learning)
[✓] 固定模型 Llama-3.2-1B-Instruct-Q4_K_M.gguf(762.81 MiB / 1.24B)
[✓] Release + Metal 构建
[✓] llama-bench 基线(PP512 / TG128,CPU vs GPU)
[✓] batch size 扫描(PP、TG,CPU vs GPU)
[✓] ngl 逐层 offload 扫描(pp512 + tg128、纯 tg 复测)
[✓] CSV 结构化输出习惯
```

**三个已经用数据确认的结论**(后续一切工作都建立在这三条上):

1. **PP 无拐点层,GPU 收益逐层均匀累积**。全 offload 869.8 t/s ≈ 纯 CPU 的 2.7 倍。16 个 decoder 层同构,GPU 对每层 matmul/attention 均匀加速。
2. **TG 在部分 offload 区间出现性能谷底**(`ngl=1~2` 时比纯 CPU 慢 23%~28%)。每 token 都要付一次 CPU↔GPU 同步固定开销,offload 层越少越亏;`ngl≥12` 后 GPU 计算量才盖过同步开销。
3. **全 offload 的 TG 是 memory-bound**。`90.5 t/s × ~0.8 GB ≈ 72 GB/s`,已进入带宽受限区(对比 M2 统一内存 ~100 GB/s 上限)。每 token 必须读一遍全部权重,这是刚需,不是某层慢。

---

# 1. 核心目标(不变)

最终目标不是一个"看完多少源码"的课程,而是一个闭环:

> **在 Apple M2 上,对 llama.cpp 的真实 LLM workload 完成一次可复现的 benchmark → profiling → bottleneck analysis → optimization → regression validation → upstream PR 闭环。**

目标分三级:

| 级别 | 目标 |
| --- | --- |
| 最低 | 建立可复现性能基线;得到 `OP → time → %` 表;深入理解一个热点 operator(MUL_MAT);完成 2~3 个实验;建立自己的 profiling/benchmark 工具 |
| 理想 | 找到一个真实性能问题;做出 5%~15% 的可重复优化;写出完整技术报告;提交 llama.cpp Issue / PR |
| 最佳 | PR 被 review / merge;GitHub 形成完整 AI Infra portfolio;简历作为核心项目 |

---

# 2. 数据给计划的三个修正(本版最重要)

原计划的主线假设是:`MUL_MAT(占 71%)→ Q4_K → Metal kernel → 优化`,并默认优化杠杆在 TG 上。**你的数据推翻了这条假设,需要三处修正。**

## 修正一:优化锚定 PP,不是 TG

- TG 全 offload 已到 ~72 GB/s,贴近带宽上限。此时 `MUL_MAT` 占大头反映的是**它必须读全部权重**,不是 kernel 写得低效。
- **打磨单个 matmul kernel 在 TG 上杠杆很小,上限就是内存带宽。**
- 真正有 kernel 优化空间的是 **PP**(compute / 并行度 bound 更明显,GPU 比 CPU 快 2.7×)。**所有 MUL_MAT + Q4_K + Metal kernel 的深挖和优化,一律在 PP workload 下做。**

## 修正二:TG 的优化方向是"减带宽",不是"改 kernel"

TG 阶段能产生真实收益的方向:

```text
更紧凑的量化          (每权重字节更少 → 每 token 读的字节更少)
KV cache 量化         (cache 占用带宽随 context 增长而变大)
算子融合              (减少中间张量往返统一内存)
批处理                (多请求共享一次权重读取 —— 注意这改变了 workload)
```

这些方向比"优化 Q4_K dequant"更符合 TG 的物理约束。判断标准统一用 roofline(见里程碑 B)。

## 修正三:ngl 谷底 = 你的第一个真实发现,现在就开始经营它

你发现的"部分 offload 时 TG 比纯 CPU 慢 23~28%"是一个**真实的、可复现的、与 M2/Metal 高度相关**的现象。它比计划里"第 10 周去网上找一个 issue"强得多:

- **P0 候选**:先用两三个反事实实验确认成因(同步开销?拷贝?命令提交?),再判断它是已知行为还是真问题。
- 若是已知行为:写一篇分析,把成因讲清楚,依然是合格的 profiling 交付物。
- 若是真问题:`-ngl` 自动选择策略(例如避开谷底区间)就是一个可以提的 upstream 改进点。

---

# 3. 主线(修正后)

```text
            你的 M2 + Llama-3.2-1B-Q4_K_M
                        │
                        ▼
          llama-bench 基线(baseline.csv / analysis.md)
                        │
                        ▼
              GGML Graph 分析(per-op)
                        │
                        ▼
               ┌────────┴────────┐
               ▼                 ▼
          PP: compute-bound   TG: memory-bound
               │                 │
               └────────┬────────┘
                        ▼
               MUL_MAT → Q4_K → Metal kernel
                        │
                        ▼
              Roofline 分析(瓶颈判断)
                        │
                        ▼
             ┌──────────┴──────────┐
             ▼                     ▼
        PP: kernel 优化         TG: 减带宽优化
             │                     │
             └──────────┬──────────┘
                        ▼
              Benchmark(mean ± stddev)
                        │
                        ▼
               Regression Matrix
                        │
                        ▼
                  Issue / PR
```

**这 10 周不再横向学习 vLLM、TensorRT、CUDA、Triton。** 你现在最需要的是完成一个闭环,不是扩展广度。

---

# 4. 排期:3 个里程碑 + 收尾(约 10 周)

剩余时间预算沿用原估算:每周 18~22h,总计 ~220h。分配原则:

```text
30% 阅读 + 30% 实验 + 25% Coding + 15% 文档
```

前几周已是数据驱动,不需要再花时间在"环境搭建"上。

---

## 里程碑 A:per-op profiling + kernel 调用链(约 2 周)

### 目标

得到你一直想要的那张表,并且能把它的每一行追到一个 kernel。

```text
PP512(全 offload,纯 GPU)

OP            TIME(ms)     %
─────────────────────────────
MUL_MAT          83.2     71.3
FLASH_ATTN       10.1      8.7
RMS_NORM          2.5      2.1
...
```

### 关键事实(已核对当前 repo)

- `llama_perf_context_data` **只有整体 timing**(`t_eval_us` 这类),没有 per-op。
- `ggml.c` 里的 `GGML_PERF` per-node 计时宏**在当前版本已不存在**。
- 所以"给 operator 加 timing"确实需要自己做——但**不要从零造**。

### 路线(从省力到深入)

1. **第一张表用 Instruments Metal System Trace 拿 per-kernel 时间**(不改代码,最快看到真相)。重点看 M2 GPU 上每个 Metal kernel 的 GPU time 和 wait time。
2. **再决定是否插桩做成可复现工具**:在 `ggml_backend_sched` / `ggml_graph_compute` 层对每个 node 计时(op 类型 + start + end + duration),输出 CSV。
3. 工具形态用 **`llama-bench` 的 CSV + 后处理脚本**,不要重写一个独立 CLI。

### 注意事项

- **`-ngl -1` 全 offload 时 per-op 表才干净**(纯 GPU 时间)。混合 offload 时同一 op 可能 CPU/GPU 都跑,表是混合的,别拿来下结论。
- 1B 模型 kernel 启动开销占比大;per-op 表 + 后续优化结论,最终都要在更大模型上复核(见里程碑 C 的模型矩阵)。

### 交付物(硬性)

```text
operator_profile.csv        ← 全 offload 下的 OP → time → %
mul_mat-call-path.md        ← llama → graph → GGML_OP_MUL_MAT → backend → kernel 完整调用链
profiling 脚本              ← 可复现:一个命令出表
```

---

## 里程碑 B:roofline 瓶颈分析(约 2 周)

### 目标

从"知道 MUL_MAT 占 71%"升级为"**用数据解释为什么它占 71%**"。

对每个 top op,建立一张表:

```text
Operator    Runtime    FLOPs    Bytes    Arithmetic Intensity    Likely bottleneck
─────────   ───────   ──────   ──────   ─────────────────────   ────────────────
MUL_MAT       71%      ...       ...           ...               memory / dequant / compute
FLASH_ATTN     9%      ...       ...           ...               memory(KV)
RMS_NORM       2%      ...       ...           ...               memory
```

### 方法(roofline 判断,直觉不发言)

1. 对每个 op 估算:
   - FLOPs(计算量)
   - 内存字节(每个权重/激活读多少字节)
   - **Arithmetic Intensity = FLOPs / Bytes**
2. 对照 M2 硬件峰值(统一内存带宽 ~100 GB/s;GPU FP16 峰值 ~5.5 TFLOPS),找到 compute-bound / memory-bound 的切换点。
3. 每个 top op 标注在 roofline 上的位置。**凡是落在 memory-bound 一侧的 op,优化方向是减字节;落在 compute 一侧的,才值得改 kernel。**

### 反事实实验(里程碑 B 的主菜)

用现成手段验证假设,而不是读代码猜:

```text
实验 B1: 全 offload vs 纯 CPU 的 per-op 对比
实验 B2: ngl 谷底成因(同步?拷贝?命令提交?)
实验 B3: 用 --override-tensor 把 attention 相关层钉回 CPU,观察瓶颈是否移动
实验 B4: 改变量化(Q4_K / Q8_0 / F16),观察 bandwidth-bound 程度变化
```

### 交付物(硬性)

```text
bottleneck-report.md      ← 每行 op 的 roofline 位置 + 结论
至少 3 个可测试的性能假设  ← 每个假设都能被一个实验证伪
ngl 谷底成因分析          ← 判断已知行为 or 真问题
```

---

## 里程碑 C:两次实验 + regression + Issue/PR(约 5 周)

### 原则(贯穿全程)

- **优化方向由里程碑 A/B 的数据决定**,不是拍脑袋。
- 每次实验:**一个假设 → 一个改动 → 一个 benchmark**,绝不混改。
- 所有实验都带 `baseline(mean ± stddev)`。

### Experiment 001(PP 侧,kernel 优化)

针对 PP workload 的 MUL_MAT kernel。候选方向:kernel 参数、threadgroup、memory layout、dequant 流水。目标 5%~15%。

### Experiment 002(TG 侧,减带宽)

针对 TG 的带宽优化方向(见修正二),或"ngl 谷底修复"。**注意**:如果 TG 的 72 GB/s 已是物理上限,预期收益要小;此时该实验的价值是"证明你理解了瓶颈",而非"必须提速"。

### Regression Matrix

优化必须在整个矩阵上验证,不许只看一个格子:

```text
                PP512   PP1024   TG32   TG128
────────────────────────────────────────────
baseline         ...
experiment1      ...
experiment2      ...
```

再加第二个维度:

```text
模型: Llama-3.2-1B(快) + Llama-3.1-8B-Q4_K_M(~4.9GB,真实 workload)
量化: Q4_K_M / Q8_0 / F16
```

**关于 8B 模型**:1B 在 M2 上 kernel 启动开销占比大、memory-bound 更极端,per-op 区分度不足。加一个 8B 模型,PP/TG 行为更接近真实部署,优化结论才有说服力。16GB 内存下 Q4_K 8B + KV cache 可以跑。

### 写 Issue(优先 ngl 谷底)

模板沿用原计划:Problem / Environment / Reproduction / Baseline / Profiling / Hypothesis / Proposed solution / Benchmark。

你的定位是"拿数据和维护者讨论性能问题",不是"新人提问"。

### 准备 PR

- 清理代码、clang-format、build、unit tests。
- 3~5 次重复 benchmark,记录 mean ± stddev。
- PR description:Motivation / Problem / Approach / Benchmark / Correctness / Compatibility / Limitations。
- 核心目标:**让维护者没有理由说你的实验不严谨**;merge 与否是加分项,不是验收线。

### 交付物(硬性)

```text
experiments/001 + 002     ← 完整实验模板
regression.csv            ← 全矩阵回归数据
upstream Issue / PR-ready patch
```

---

## 第 12 周:Portfolio 整理

不再改代码。把项目整理成 GitHub 资产(结构见第 10 节),README 第一屏直接放:

```text
Hardware     Apple M2 / 16GB
Backend      Metal
Model        Llama-3.2-1B / Llama-3.1-8B (Q4_K_M)
Baseline     PP512: XX tok/s   TG128: XX tok/s
Optimized    PP512: YY tok/s   TG128: ZZ tok/s
Improvement  PP512: +X%        TG128: +Y%
```

---

# 5. 关键方法论(贯穿整个项目)

1. **Performance measurement ≠ 一个数字。** 永远 `120.3 ± 0.8 tok/s`,不要 `120 tok/s`。所有实验至少重复 3~5 次。
2. **一次只改一个变量。** 不许"改了 A、B、C 然后快了"——你不知道为什么变快,等于没做。
3. **roofline 说话,直觉不发言。** 每个"瓶颈"都要落在"带宽还是算力"二选一上,否则它只是形容词。
4. **全 offload 下做 per-op。** 混合 offload 的表是 CPU/GPU 混合的,别拿来下结论。
5. **局部提升 ≠ 优化。** 只让 PP512 快 10% 但 TG128 慢 15% 的改动,不是优化。
6. **数据归档。** 每个实验存原始 CSV,不截图、不手抄。

---

# 6. 学习资料优先级(微调)

| 优先级 | 内容 |
| --- | --- |
| P0 | llama.cpp 当前源码:ggml/src、ggml/src/ggml-metal、tools/llama-bench;遇到问题按需追,不系统通读 |
| P1 | Apple 官方:Metal 编程指南、Xcode Instruments(Metal System Trace)、Apple GPU profiling |
| P2 | Q4_K / quantization 布局 |
| P3 | Attention / KV cache(为 TG 带宽优化做铺垫) |
| 暂时不碰 | CUDA / Triton / vLLM / TensorRT / DeepSpeed / Megatron |

**学习方式:problem driven,不是 source-code driven。** 不要再"今天看 scheduler,明天看 backend"。

```text
为什么 MUL_MAT 占 71%?
       ↓
需要看 scheduler?看。
需要看 backend?看。
需要看 Metal?看。
需要看 Q4_K?看。
```

---

# 7. 硬性验收标准(里程碑级)

不要用"这周看了很多源码"衡量自己,用产物衡量。

| 里程碑 | 必须有 | 否则意味着 |
| --- | --- | --- |
| A | `operator_profile.csv` + `mul_mat-call-path.md` + 可复现脚本 | profiling 能力未建立,不进入优化 |
| B | `bottleneck-report.md` + ≥3 个可测试假设 + ngl 谷底成因结论 | 没有性能模型,优化是瞎撞 |
| C | `experiments/001` `002` + `regression.csv` + upstream Issue / PR-ready patch | 闭环未完成,不算项目结束 |
| 收尾 | GitHub repo + 技术报告 + benchmark 数据 + profiling 工具 + patch | — |

---

# 8. 如果卡住

**最多允许连续两天卡在同一个问题上。** 两天没进展就改打法:

```text
写最小实验
 ↓
简化 workload
 ↓
验证一个假设
```

目标不是"把所有源码理解",而是**解决一个性能问题**。

---

# 9. Android 手机:第 12 周之前不加入

这不是"以后再说",是明确的取舍。你已经同时有 SLAM / SfM / 导航 / GGML / llama.cpp / KV Cache / Android / AI Infra 多条线,再加 Android(→ ARM / Vulkan / NNAPI / OpenCL)会迅速变成"什么都懂一点,没有一个完整成果"。

第一个项目完成后的第二阶段,再把它扩展成:

```text
Cross-platform LLM inference performance analysis
on Apple Silicon and ARM mobile devices
```

那时你的 M2/Metal 经验可以完整迁移到 Android/ARM/NEON。

---

# 10. 最终 GitHub 结构

```text
llama-cpp-performance-lab
│
├── README.md                ← 第一屏:机器 / workload / baseline / 优化 / improvement
├── benchmark/
│   ├── baseline/
│   ├── pp/ tg/ regression/
├── profiler/
│   ├── operator/ scripts/
├── experiments/
│   ├── 001/ 002/
├── analysis/
│   ├── workload.md  mul_mat.md  q4_k.md  metal.md
├── patches/
└── reports/
    └── final-report.md
```

面试官点进去看到的是一条完整的证据链:

```text
我的机器 → workload → benchmark → profiling → bottleneck → hypothesis → patch → improvement → regression → upstream PR
```

这就是一个完整的 AI Infra 性能工程项目,不是"我学习了 llama.cpp"。

---

# 11. 协作方式

第 3~4 周的里程碑 A(per-op profiling + MUL_MAT → Metal kernel 路径)建议作为第一个阶段性项目一起推进:

```text
你跑数据 / 改代码
 ↓
给我原始输出(CSV、日志)
 ↓
一起分析,定下一个实验
```

拿到这组 baseline 后,下一步非常具体:**"如何在 llama.cpp 的 ggml_backend_sched / graph_compute 层给 operator 插桩计时,并最终生成 `MUL_MAT 71.3% / FLASH_ATTN 8.7%` 这张表。"**

这一步做出来,整个项目就真正启动了。
