# PP 阶段 MUL_MAT 瓶颈:最终结论技术报告(M2)

> **报告性质**:里程碑 C 完整分析链(Experiment 001→002→003)的最终归因交付 | 2026-09-02
> **硬件**:Apple M2(8 核 GPU,统一内存 ~100 GB/s)
> **模型**:Llama-3.2-1B-Instruct-Q4_K_M(及其 F16 / Q8_0 转档对照)
> **工具链**:`llama-bench`(整图基准)、`examples/gemm-bench`(隔离 GEMM 微基准)、`examples/mma-probe`(合成 MMA 指令探针)
> **过程日志**:[analysis.md](./analysis.md) | **Experiment 001 深度报告**:[report-pp-mul-mat.md](./report-pp-mul-mat.md)
> **阅读目标**:这份报告回答最初的问题——"PP(提示词处理)下 MUL_MAT 为什么只有 ~38% 算力利用率、还有没有优化空间",并给出可复现的证据链。

---

## 摘要:问题与最终答案

**问题**:在这个 M2 上跑 llama.cpp 的 PP(Prompt Processing,预填充/提示词处理)阶段,一次前向计算里 MUL_MAT(Matrix Multiplication,矩阵乘法)占 ~87% 的时间;但它的算力利用率只有 ~30-38%,远低于 5.8 TFLOPS 的理论峰值假设。为什么?kernel(内核)优化还有没有空间?

**最终答案(一句话)**:**PP 的 MUL_MAT 是 compute-bound(算力受限),但"利用率低"的主因不是反量化、不是同步、也不是访存编排——而是这台 M2 没有 tensor core(tensor 核心/专用矩阵硬件),它实际走的是 Apple 8×8 simdgroup(单指令多数据组)fp16 矩阵乘指令路径,该指令本身的硬件可达吞吐只有 ~2.8 TFLOPS(对 5.8 TF 假设约 48%)。ggml 的 legacy kernel(旧式内核)已经跑到了这条指令路径硬件上限的 ~88%,PP kernel 级优化在此硬件上已无大的空间。**

**顺带的发现**:**5.8 TFLOPS 的 FP16 峰值假设,对"8×8 simdgroup fp16 指令"这条路不成立。** M2 的该路径实测上限 ~2.8 TF,更接近 Apple 公布的 FP32 数值(~3.6 TF)——Apple GPU 的 fp16 不提供双倍吞吐。若以 3.6 TF 计,ggml kernel 实际已达硬件的 ~68%。

---

## 一、背景速览

### 1.1 为什么只研究 MUL_MAT

LLM(Large Language Model,大语言模型)推理分 PP(Prompt Processing,预填充)与 TG(Token Generation,逐词生成)两阶段。PP 一次性处理整段提示词,**计算量大、是 compute-bound(算力受限)**;TG 逐 token 生成,**每次只算一个 token、是 memory-bound(内存受限)**。profiling 显示 PP512 下一次前向里 MUL_MAT 占 87.3%(见 [report-pp-mul-mat.md](report-pp-mul-mat.md) 第三节)——**动 PP 的速度,几乎只能动 MUL_MAT。**

### 1.2 为什么"利用率低"需要严谨归因

roofline(屋顶线)分析算出所有大 MUL_MAT 的 AI(Arithmetic Intensity,算术强度)都在 360~1500 FLOPs/byte,远超脊点 58 → 理论上是深度 compute-bound(见 analysis.md Roofline 节)。但"compute-bound"只回答"瓶颈在算力",没回答"**算力为什么没跑满**"。后一个问题有几个候选答案,它们导致**完全不同的优化方向**,所以必须用实验逐个证伪,不能拍脑袋。

### 1.3 分析链总览

```
Experiment 001  反量化是主因吗?      → 不是,dequant 只占 ~11%(探针证伪)
Experiment 002  kernel 是嵌在图里被拖累的吗? → 隔离单跑也只有 41-43%,墙是 kernel 自身的
Experiment 003  墙是同步?             → 不是,删 barrier 免费(003-S1)
Experiment 003  墙是 kernel 编排?     → 只剩 ~5pp;纯 MMA 指令本身只有 46-48%(003-S2)
最终归因        M2 8×8 fp16 指令路径硬件上限 ~2.8 TF → ggml 已达其 ~88%
```

---

## 二、核心测量数据(全部在同机复现)

| 测量 | 值 | 来源 |
|---|---|---|
| llama-bench Q4_K pp512 | 994.2 ± 1.3 tok/s | 无插桩整图 |
| llama-bench F16 pp512 | 1208.4 ± 4.0 tok/s(+21.5%) | 同 |
| Q4_K MUL_MAT 绝对时间 | 1028.8 ms(占 graph 87.3%) | LLAMA_OP_PROFILE |
| F16 MUL_MAT 绝对时间 | 834.7 ms | 同 |
| **隔离单 F16 kernel(FFN 层形状)** | **41.4%(2.40 TFLOPS)** | gemm-bench |
| 隔离单 kernel 极限形状(K4096 N8192 M4096) | 43.1%(平台) | gemm-bench |
| **合成纯 MMA 指令流(无访存/同步)** | **46-48%(2.7-2.8 TFLOPS)** | mma-probe |
| ggml kernel 相对纯 MMA 硬件墙 | 41-43% / 48% ≈ **88%** | 比值 |

注:整图 F16 ≈ 38%,隔离单 kernel 41.4% → 图调度/同步税只有 ~3-5 个百分点;隔离也上不去,墙在 kernel 内部。

---

## 三、Experiment 001:反量化不是主因(成本分解)

> 完整过程见 [report-pp-mul-mat.md](report-pp-mul-mat.md) 与 analysis.md Experiment 001。

**问**:Q4_K(4-bit 量化权重)要反量化才能算,Q4_K 的 MUL_MAT 比 F16 慢 23.3%——是反量化计算拖的吗?

**三点对照**(同一模型转档,权重字节数递增):Q8_0(8-bit 极简反量化)比 Q4_K 字节多 64%,反而快 10%;F16 字节最多,反而最快。→ 排序由反量化复杂度而非带宽决定。

**kernel 探针**(把 `dequantize_func` 反量化计算跳过、保留读取防优化删读):Q4_K 1028.8 → 912.1 ms(−11.3%),与 Q8_0 对照组(923.7 ms)交叉印证。

**成本分解**(同一 profile 口径下的减法,Q4_K MUL_MAT = 1028.8 ms):

| 组成 | 计算 | 耗时 | 占比 |
|---|---|---|---|
| F16 基线(无反量化、规整访存) | 直接测 | 834.7 ms | 81.1% |
| dequant(Dequantization,反量化)计算 | 1028.8 − 912.1 | 116.7 ms | 11.3% |
| 4-bit 权重解包 / kernel 分支 | 912.1 − 834.7 | 77.4 ms | 7.5% |

**本节结论**:**反量化只占 ~11%,不是"利用率 30% 的原因"。** 优化双缓冲、抠反量化分支,理论天花板就是这 ~11%,且要与共享内存布局深度耦合、收益低。**真正要解释的是 F16 基线那 834.7 ms——纯矩阵乘路径也只有 ~38%。** 分析重心转移到这个"38% 的墙"上。

---

## 四、Experiment 002:墙是 kernel 自身的,不是整图拖累

> 过程与工具见 analysis.md Experiment 002 与 [examples/gemm-bench/gemm-bench.cpp](../../examples/gemm-bench/gemm-bench.cpp)。

**问**:38% 是在整张 GGML 计算图里测的——有调度、同步、前后算子。如果把单个 MUL_MAT 摘出来纯跑,是不是能高得多?

**方法**:新增独立微基准 `llama-gemm-bench`:把 workload 压成一个纯 F16 GEMM(General Matrix Multiply,通用矩阵乘),只由 Metal backend 单独调度这一个 op;形状对齐 Llama-3.2-1B 真实层,再扫 M(批次)放大。配 `--verify`(Metal vs CPU 数值对照)兜底正确性。

**结果**:

| 形状 | 利用率 |
|---|---|
| FFN up(2048×8192)× M=512(模型实际形状) | **41.4%** |
| FFN down / attn proj(M=512) | 41.0% / 41.5% |
| M=1024 → 4096(批次放大) | 42.4% → 43.2% |
| K4096 N8192 M4096(极限,单 dispatch 275 GFLOP) | 43.1%(平台) |

**本节结论**:
1. **隔离单 kernel = 41.4%,整图 ≈ 38%** → 图调度/同步税只有 ~3 个百分点,不是主因。
2. **M 一路放到 4096、K 放到 4096,利用率停在 43% 平台** → 不是任务量/occupancy(驻留线程数)不足。
3. **43% 是纯 F16、无任何反量化** → 墙在矩阵乘指令的发射结构本身(8×8 simdgroup 指令的利用率 + 每 K-tile 的编排),与量化无关。

---

## 五、Experiment 003:墙的归因(同步 vs 硬件)

### 5.1 003-S1:删除 PHASE2 的 simdgroup barrier——同步免费

**问**:legacy kernel(旧式内核)每个 K-tile 迭代带 2 个全组 `threadgroup_barrier` + 12 个 `simdgroup_barrier`,同步是不是墙的一部分?

**方法**:删掉 `mul_mm.metal` legacy 分支 PHASE2 里每个 ik 迭代的 3 处 `simdgroup_barrier(mem_flags::mem_none)`(仅 `kernel_mul_mm`,不动 `kernel_mul_mm_id`),跑 `--verify` 确认数值 PASS 后测速。

**结果**:

| 形状 | baseline | S1 | 变化 |
|---|---|---|---|
| FFN up 2048×8192×512 | 41.4% | 40.9% | −0.5pp(噪声内) |
| attn proj 2048×2048×512 | 41.5% | 40.8% | −0.7pp(噪声内) |
| saturate 4096×8192×4096 | 43.1% | 42.6% | −0.5pp(噪声内) |

**本节结论**:**同步指令 ≈ 免费,删掉没有任何正收益。** 量化估算一致:barrier 即使按 ~200 cycle 计,也占 threadgroup 生命周期(K=2048 时 ~9.4M cycle)不到 1%。→ 顺带证伪了"NK 翻倍摊薄固定开销"的收益来源——它主要省的就是 barrier,已证明不占关键路径。

### 5.2 003-S2:合成 MMA 探针——直接测硬件指令上限

> 工具见 [examples/mma-probe/mma-probe.mm](../../examples/mma-probe/mma-probe.mm)。

**问**:43% 的墙,是 ggml kernel 编排浪费,还是 Apple 8×8 simdgroup fp16 矩阵乘指令本身的吞吐上限?

**方法**:写一个与 ggml 完全解耦的最小 Metal kernel:一对 8×8 fp16 矩阵 A、B 直接填进寄存器,循环 iters 轮只发 `simdgroup_multiply_accumulate`(SIMD 组乘累加)指令——**没有 smem(shared memory,共享内存)、没有 load、没有 barrier**。结果只 store 一次防死代码消除,内置线性度自检(iters 与 2×iters 时间比应 ≈2)排除编译器优化掉循环。

**结果**:

| 变体 | threads | ILP(指令级并行) | TFLOPS | util%(对 5.8) | 线性度 |
|---|---|---|---|---|---|
| 小计算量(无效,launch 主导) | 8192 | 4 | 0.81 | 13.9% | 0.93(作废) |
| 大计算量 | 32768 | 4 | **2.71** | 46.7% | 1.98 ✓ |
| 更多线程(occupancy 上升) | 65536 | 4 | 2.68 | 46.1% | 1.99 ✓ |
| ILP 翻倍(8 个独立累加器) | 32768 | 8 | **2.78** | 48.0% | 2.00 ✓ |

**本节结论**:**纯 8×8 fp16 MMA 指令流(零访存、零同步、零反量化)的硬件上限 ≈ 2.7-2.8 TFLOPS,即对 5.8 TF 假设的 ~48%。** 加线程不涨、ILP 4→8 只涨 1.3pp → 排除了"探针自己没写满"。这是指令吞吐级的天花板,不是资源调度问题。

---

## 六、最终归因

把三段证据拼起来:

```
纯 8×8 fp16 MMA 指令(硬件,探针)   : 46-48%  (~2.7-2.8 TF)   ← 指令路径的真实天花板
ggml F16 legacy kernel(隔离)      : 41-43%  (~2.4-2.5 TF)
                                      │
                                      └─ 差 ~5pp = PHASE2 的 smem→寄存器 load 编排税
                                          (每推进 8 层 K 需 4+2 条 load 喂 8 条 MMA,
                                           8×8 指令模型的硬约束,几乎不可再削)
ggml F16 kernel(整图 llama-bench)  : ~38%    ← 再减 ~3-5pp 图调度/同步税
Q4_K kernel(整图)                  : ~31%    ← 再减 dequant ~11% + 解包 ~7%
```

**三层结论**:

1. **M2 无 tensor core(`has_tensor=false`),走 8×8 simdgroup fp16 指令路径,这条路径的硬件可达上限只有 ~2.8 TF(48%)。** 这是全部分析里最底层的一堵墙——比它再低的任何 kernel 利用率都在这堵墙之内解释。
2. **ggml 的 legacy kernel 已到这堵墙的 ~88%**(2.45 / 2.78)。剩余 ~5pp 是 8×8 指令模型固有的 load 编排税 + ~3-5pp 图调度税,都难以实质压缩。**ggml 在这台 M2 的 PP 路径上已经接近它所在指令路径的物理上限。**
3. **5.8 TFLOPS 的 FP16 峰值假设对 8×8 fp16 路径不成立**,实测上限更接近 Apple 的 FP32 数值(~3.6 TF)。若以 3.6 TF 为分母,ggml kernel 已达硬件 ~68%、纯 MMA ~77%——"利用率只有三成"很大程度上是**用了对不上该指令路径的峰值假设**造成的视觉偏差。

**归因一句话**:PP MUL_MAT 的"低利用率" ≈ 48% 的硬件指令墙(纯 MMA)+ ~5pp 访存编排税 + ~3-5pp 图调度税;反量化(001)、同步(003-S1)都不是瓶颈,访存编排只占 ~5pp(003-S2)。**M2 legacy kernel 已接近其指令路径的物理上限,在此硬件上做 PP kernel 优化没有大的空间。**

---

## 七、对优化方向的指导

| 方向 | 预期收益 | 结论 |
|---|---|---|
| 优化 `dequantize_q4_K` / 双缓冲 | ≤11%(已实测天花板) | **不值得作首选**,成本高、与格式耦合 |
| NK 翻倍(32→64) | ≈0 | 其收益来源(摊薄 barrier/装货)已被 003-S1 证伪 |
| 压缩 load 编排税 | ≤5pp | 8×8 指令模型硬约束,几乎不可削 |
| **突破硬件墙** | — | 需 tensor core 路径(M3/A 系列支持 tensor API)或换机,不是改 legacy kernel 能解决的 |
| **转向 TG 侧** | 方向明确 | TG 是 memory-bound,减带宽是真实可做的下一站 |

**建议**:PP 侧的 kernel 级优化在此硬件上接近收益耗尽。若要继续提性能,方向应转向:(a) TG 阶段的带宽优化(此前的 `-ngl` 扫描已发现部分 offload 性能谷底问题,方向明确);(b) 若必须压 PP,评估带 tensor 支持的硬件路径(M3 及以上的 tensor API 走的是另一套 kernel,不在此结论范围内)。

---

## 八、方法学沉淀:为什么这套结论可信

1. **每一层"墙"都用独立工具隔离测量**:整图(llama-bench)→ 单 kernel(gemm-bench)→ 纯指令(mma-probe),层层剥离,每一步回答一个可证伪的问题。
2. **探针给出的是"上限",用上限决策**:把反量化完全跳过得 11.3%,把访存/同步完全去掉得 48%——"它最多值这么多"比猜可靠。
3. **每个 kernel 改动都过 `--verify`(Metal vs CPU 数值对照)**才谈性能,防止"改快了但是算错了"。
4. **发现了"不成立的峰值假设"**:纯 MMA 探针(46-48%)反推出 5.8 TF 对 8×8 fp16 路径过高——没有探针,会一直用错误的分母得出"ggml 很烂"的误判。
5. **结论对假设的依赖被显式标注**:41-43% vs 48% 的相对关系不依赖任何峰值假设;只有绝对 util% 依赖(且该假设本身已被实验修正)。

---

## 附录 A:复现命令

```bash
# ① 整图基准(pp512,插桩拿 MUL_MAT 绝对时间)
./build/bin/llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 32 -r 2 -ngl -1
LLAMA_OP_PROFILE=1 ./build/bin/llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 0 -r 1 -ngl -1

# ② 隔离单 F16 kernel(微基准;--verify 先做数值校验)
cmake --build build --target llama-gemm-bench
./build/bin/llama-gemm-bench --verify
./build/bin/llama-gemm-bench

# ③ 合成纯 MMA 探针(线性度自检应 ≈2)
cmake --build build --target llama-mma-probe
./build/bin/llama-mma-probe 32768 200000

# ④ 003-S1(可选复现):删 mul_mm.metal legacy 分支 PHASE2 的三处
#    simdgroup_barrier(mem_flags::mem_none) 后重编,再跑 ②
```

## 附录 B:关键数字速查

| 数字 | 值 |
|---|---|
| Q4_K pp512 MUL_MAT | 1028.8 ms |
| F16 pp512 MUL_MAT | 834.7 ms(~38%) |
| dequant 计算成本 | 116.7 ms(11.3%,实测上限) |
| 4-bit 解包/分支 | 77.4 ms(7.5%) |
| 隔离单 F16 kernel | 41.4%(模型形状)→ 43% 平台 |
| 纯 8×8 fp16 MMA 指令 | 46-48%(2.7-2.8 TF) |
| ggml kernel / 硬件墙 | ~88% |
| M2 该指令路径实测峰值 | ~2.8 TF(≈ FP32 数值 3.6 之下) |

---

*本报告为 Experiment 001(深度分析见 [report-pp-mul-mat.md](report-pp-mul-mat.md))→ 002(隔离 GEMM 微基准)→ 003(barrier 实验 + 合成 MMA 探针)三段实验的最终归因。全部原始数据、命令与 kernel 改动记录见 [analysis.md](./analysis.md)。*
