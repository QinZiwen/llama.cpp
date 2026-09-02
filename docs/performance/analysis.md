# gpu计算
./bin/llama-bench -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl -1 

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |        968.59 ± 5.41 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg128 |         91.39 ± 0.56 |

# cpu计算
./bin/llama-bench -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 0  

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |        316.13 ± 2.13 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg128 |         70.42 ± 2.27 |

# 测试 batch size 对 PP 的影响

## cpu

```bash
./bin/llama-bench \
  -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p 1,8,32,64,128,256,512,1024 \
  -n 0 \
  -r 3 \
  -ngl 0
```

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             pp1 |         72.47 ± 1.02 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             pp8 |        260.99 ± 3.07 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            pp32 |       255.91 ± 66.26 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            pp64 |        323.77 ± 3.39 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp128 |        311.12 ± 4.65 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp256 |        295.22 ± 0.09 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |        300.40 ± 9.13 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |          pp1024 |       279.57 ± 13.38 |

## gpu

```bash
./bin/llama-bench \
  -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p 1,8,32,64,128,256,512,1024 \
  -n 0 \
  -r 3 \
  -ngl -1
```

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             pp1 |         88.51 ± 5.64 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             pp8 |        197.75 ± 0.90 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            pp32 |       800.73 ± 19.48 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            pp64 |       934.40 ± 10.50 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp128 |        953.60 ± 3.31 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp256 |       971.30 ± 13.42 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |        966.64 ± 8.53 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |          pp1024 |       938.65 ± 12.63 |

# 测试 batch size 对 TG 的影响

## cpu

```bash
./bin/llama-bench -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 1,8,32,64,128,256,512,1024 -r 3 -ngl 0
```

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |       307.76 ± 16.76 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             tg1 |         65.06 ± 1.67 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             tg8 |         69.98 ± 1.69 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            tg32 |         67.92 ± 3.05 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            tg64 |         71.27 ± 2.45 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg128 |         70.22 ± 1.07 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg256 |         69.89 ± 2.03 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg512 |         68.44 ± 0.72 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |          tg1024 |         62.79 ± 1.89 |

# gpu

```bash
./bin/llama-bench -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 1,8,32,64,128,256,512,1024 -r 3 -ngl -1
```

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           pp512 |       991.49 ± 16.11 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             tg1 |         83.35 ± 5.82 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |             tg8 |         94.07 ± 0.96 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            tg32 |         94.25 ± 0.50 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |            tg64 |         94.12 ± 0.10 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg128 |         88.59 ± 2.82 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg256 |         93.82 ± 0.26 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |           tg512 |         91.91 ± 1.05 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | BLAS,MTL   |       4 |          tg1024 |        76.35 ± 11.75 |

# ngl 逐层 offload 扫描（定位差异来源）

目的：llama-bench 只看 pp/tg 聚合值，粒度太粗。通过把模型一层一层从 CPU 移到 GPU（`-ngl` 多值扫描），观察每加一层时 pp/tg 吞吐的变化，定位性能差异来自哪些层/阶段。

## 命令

```bash
# 扫描 1（pp + tg 两组，各 -ngl 多值）
./bin/llama-bench \
  -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 -r 1 \
  -ngl 0,1,2,4,8,12,14,15,16 \
  -o csv > build/ngl_scan.csv

# 扫描 2（`-p 0` 只测纯生成，无 prompt 处理干扰，复测 TG 曲线）
./bin/llama-bench \
  -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p 0 -n 128 -r 1 \
  -ngl 0,1,2,4,8,12,14,15,16 \
  -o csv > build/ngl_scan_2.csv
```

Llama-3.2-1B 共 16 层，`-ngl` 可扫到每个层。CSV 中 `n_prompt=512` 行为 pp512、`n_gen=128` 行为 tg128；扫描 2 因 `-p 0` 只有 tg 行（`avg_ts` 为 t/s）。

## 扫描 1 结果（pp512 + tg128）

| ngl | pp512 (t/s) | 相对 CPU | tg128 (t/s) | 相对 CPU |
|----:|------------:|----------:|------------:|----------:|
|   0 |      323.7  |     基准 |      71.1  |     基准 |
|   1 |      320.5  |     −1%  |      54.7  |   −23%  |
|   2 |      313.4  |     −3%  |      51.2  |   −28%  |
|   4 |      348.9  |     +8%  |      59.3  |   −17%  |
|   8 |      423.8  |    +31%  |      64.1  |   −10%  |
|  12 |      572.0  |    +77%  |      77.7  |    +9%  |
|  14 |      689.1  |   +113%  |      87.4  |   +23%  |
|  15 |      761.7  |   +135%  |      88.7  |   +25%  |
|  16 |      869.8  |   +169%  |      90.5  |   +27%  |

## 扫描 2 复测（只测 tg128，趋势一致）

| ngl | tg128 (t/s) |
|----:|------------:|
|   0 |      64.4  |
|   1 |      56.2  |
|   2 |      50.7  |
|   4 |      50.2  |
|   8 |      56.0  |
|  12 |      64.2  |
|  14 |      73.5  |
|  15 |      75.2  |
|  16 |      91.8  |

单次测量（r=1）噪音较大（ngl=0 两次分别为 71.1 / 64.4），但 TG 曲线形状一致：部分 offload 区间（ngl=1~8）显著变慢，全 offload 最快。

## 结论

1. **PP 无拐点层，收益逐层均匀累积**：pp512 从 ngl=0→16 单调上升（除前 1~2 层因引入同步略有回退），16 个 decoder 层同构，GPU 对每层 matmul/attention 加速均匀。全 offload 869.8 t/s ≈ CPU 的 2.7 倍。
2. **TG 在部分 offload 区间出现性能谷底**（ngl=1~2 时比纯 CPU 慢 23%~28%）：生成逐 token 执行，每 token 都要做一次 CPU↔GPU 往返（Metal 命令提交 + 跨 backend 同步 + 拷贝），该固定延迟在 pp 阶段被 batch 内 token 摊平，在 tg 阶段则每 token 付一次全款；offload 层越少，摊到的 GPU 计算越少，越亏。ngl≥12 后 GPU 计算量才盖过同步开销。
3. **即使全 offload，TG 也只比 CPU 快 27%**：90.5 t/s × 800 MB ≈ 72 GB/s，已贴近 Apple M2 统一内存带宽上限。TG 是 memory-bound，差异根源是"每 token 必须读一遍全部权重"，而非某层慢。

## 实践建议

- 生成（tg）任务：**避免 `-ngl 1~4`**（性能最差区间）。要么 `-ngl -1` 全 offload，要么 `-ngl 0` 纯 CPU。
- 预填充（pp）任务：无脑全 offload 最优。
- 若需继续深挖 kernel 级差异：针对全 offload 的 tg 场景用 Instruments Metal System Trace 抓每个 Metal kernel 耗时与同步等待；或用 `--override-tensor` 把 attention 相关层钉回 CPU 做反事实实验。

---

# GGML Operator Profiling（里程碑 A 第一步，2026-08-30）

## 方法

在 `src/llama-context.cpp` 加了实验性 per-op 计时，通过环境变量开关：

```bash
LLAMA_OP_PROFILE=1 ./bin/llama-bench -m ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 0 -r 1 -ngl -1
```

原理：复用 `llama_context_params.cb_eval` 的 eval callback 机制。callback 对每个 node 返回 `need=true` 时，`ggml_backend_sched` 会把每个 node 单独提交 + 同步，从而在 `ask=true`（提交前）/ `ask=false`（完成后）之间测到该 op 的一次执行时间。

**已知局限（读表前必看）**：
- 开启后每个 node 被强制同步，性能显著下降（pp512 从 ~870 掉到 ~811 t/s）。
- 测到的是"提交 + 执行 + 同步"的墙钟时间，**绝对时间偏大**；尤其 `RESHAPE / VIEW / PERMUTE` 这类纯元数据/视图 op，真实执行时几乎零成本，但在这里被同步开销撑大。
- 因此本表用于看 **op 之间的相对占比趋势**，不能当真实性能；绝对数值以 `llama-bench` 的 tok/s 为准。

## PP512 全 offload（n_prompt=512, n_gen=0）

```
OP                            calls     time(ms)        %
-----------------------------------------------
MUL_MAT                         226     1049.392   79.18%
FLASH_ATTN_EXT                   32       50.484    3.81%
RESHAPE                         128       35.028    2.64%
VIEW                            160       32.562    2.46%
GLU                              32       29.077    2.19%
ADD                              64       25.648    1.94%
RMS_NORM                         66       21.579    1.63%
PERMUTE                          96       20.514    1.55%
SET_ROWS                         64       19.807    1.49%
MUL                              66       18.825    1.42%
ROPE                             64       18.448    1.39%
GET_ROWS                          6        3.928    0.30%
-----------------------------------------------
TOTAL                                   1325.292  100.00%
```

## TG32 全 offload（n_prompt=0, n_gen=32）

```
OP                            calls     time(ms)        %
-----------------------------------------------
MUL_MAT                        3729     1010.147   32.10%
VIEW                           2640      427.406   13.58%
RESHAPE                        2112      341.665   10.86%
PERMUTE                        1584      254.999    8.10%
ADD                            1056      191.343    6.08%
ROPE                           1056      184.076    5.85%
RMS_NORM                       1089      179.866    5.72%
MUL                            1089      177.320    5.63%
SET_ROWS                       1056      175.140    5.57%
GLU                             528      102.009    3.24%
FLASH_ATTN_EXT                  528       92.481    2.94%
GET_ROWS                         99       10.703    0.34%
-----------------------------------------------
TOTAL                                   3147.155  100.00%
```

## 过滤元数据 op 后重测（2026-08-30）

把 `RESHAPE / VIEW / PERMUTE / GET_ROWS / SET_ROWS` 从统计中剔除后重跑（改 `src/llama-context.cpp` 过滤逻辑后重新编译，同一命令）。

### PP512（过滤后）

```
OP                            calls     time(ms)        %
-----------------------------------------------
MUL_MAT                         226     1040.664   86.46%
FLASH_ATTN_EXT                   32       50.167    4.17%
ADD                              64       26.712    2.22%
GLU                              32       26.703    2.22%
RMS_NORM                         66       23.506    1.95%
MUL                              66       18.503    1.54%
ROPE                             64       17.377    1.44%
-----------------------------------------------
TOTAL                                   1203.632  100.00%
```

### TG32（过滤后）

```
OP                            calls     time(ms)        %
-----------------------------------------------
MUL_MAT                        3729      979.300   51.80%
ADD                            1056      187.463    9.92%
RMS_NORM                       1089      183.475    9.70%
MUL                            1089      177.207    9.37%
ROPE                           1056      165.634    8.76%
GLU                             528      100.323    5.31%
FLASH_ATTN_EXT                  528       97.193    5.14%
-----------------------------------------------
TOTAL                                   1890.595  100.00%
```

### 失真程度验证

| workload | MUL_MAT 原始占比 | 过滤后占比 | 元数据 op 同步税占比 |
| --- | --- | --- | --- |
| PP512 | 79.18% | **86.46%** | 122ms / 1325ms ≈ 9.2% |
| TG32 | 32.10% | **51.80%** | 1257ms / 3147ms ≈ 39.9% |

结论：

- **PP 失真小（约 7 个百分点）**：元数据 op 在 PP 中本来就少，过滤后 MUL_MAT 从 79% → 86.5%，仍是压倒性热点。
- **TG 失真巨大（约 20 个百分点）**：TG 每 token 都跑一遍全部 op，元数据 op 数量暴增（原始表里 VIEW/RESHAPE/PERMUTE 占 32.5%），几乎全是同步税。过滤后 MUL_MAT 从 32% → 51.8%。
- **过滤后 TG 仍无单一 dominant op**：MUL_MAT 过半，但 ADD/RMS_NORM/MUL/ROPE 各占 ~9%，时间被均匀分摊 —— 正是 memory-bound（都在等带宽）的特征。

## 对比与洞察

1. **MUL_MAT 是唯一 dominant op**：PP512 占 79.2%，TG32 占 32.1%。这与原计划假设的 `MUL_MAT ~71%` 一致，PP 场景下 MUL_MAT 就是绝对热点。

2. **TG 下 MUL_MAT 占比骤降，但这不是它变快了**：TG 每次只喂 1 个 token，graph 节点数暴增（MUL_MAT calls：PP 226 → TG 3729），所有 op 都每 token 跑一遍。TG 里 `VIEW+RESHAPE+PERMUTE` 从 PP 的 6.7% 涨到 32.5% —— 这几乎全是 profiling 的 per-node 同步开销，真实执行时它们是 ~0。

3. **排除元数据 op 后（实测过滤，见上节）**：MUL_MAT 占比 PP512 79.2% → **86.5%**，TG32 32.1% → **51.8%**。与上文的粗估（≈84.8% / ≈47.6%）方向一致，但实测值更高，说明同步税比预想更大。

   PP 场景 MUL_MAT 是一边倒的热点；TG 场景其他计算 op（ADD/ROPE/RMS_NORM/MUL/GLU/FLASH_ATTN_EXT）合计仍占 ~48%，没有单一 dominant op —— 这与"TG 是 memory-bound、所有 op 都在等内存"的结论互相印证。

4. **下一步**：
   - 用 roofline 判断：PP 下 MUL_MAT（占 86.5%）是 compute-bound 还是 memory-bound —— 决定 kernel 优化空间（里程碑 B）；
   - 若要干净的真实时间（不含同步税），需要去掉强制同步的插桩方式（后续里程碑 B 再做）。

---

# Roofline（屋顶线）分析：PP 下 MUL_MAT 是 compute-bound（计算受限）（里程碑 B，2026-08-30）

## 硬件参数（Apple M2 Air，8 核 GPU）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| GPU | Apple M2，**8 核** | `system_profiler SPDisplaysDataType` 确认 |
| 统一内存带宽 | **~100 GB/s** | M2 规格 |
| FP16 理论峰值 | **~5.8 TFLOPS** | 8 核 GPU（FP16 = FP32×2；10 核版为 7.2） |
| **Ridge Point（脊点）** | **~58 FLOPs/byte** | 5.8e12 / 100e9 |

roofline（屋顶线）判断规则：`AI（Arithmetic Intensity，算术强度）> ridge` 为 compute-bound（计算受限），`AI < ridge` 为 memory-bound（内存受限）。

## MUL_MAT 形状与 Arithmetic Intensity（AI，算术强度）（PP512，形状来自 profiling 真实数据）

一次 graph 的 MUL_MAT 明细（profiling 表内 calls 为 warmup+正式两次，除以 2 得单次）：

| 权重类型 | 形状（W × A） | 单次 graph calls | FLOPs/次 | bytes/次 | **AI（算术强度）**（FLOPs/byte） | MM 时间占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q4_K | 2048×8192 × 2048×512 | 30 | 17.2e9 | 11.5e6 | **1489** | 51.8% |
| q4_K | 2048×2048 × 2048×512 | 32 | 4.3e9 | 4.5e6 | 964 | 15.7% |
| q4_K | 8192×2048 × 8192×512 | 8 | 17.2e9 | 17.8e6 | 964 | 13.9% |
| q6_K | 8192×2048 × 8192×512 | 7 | 17.2e9 | 22.2e6 | 776 | 12.5% |
| q4_K | 2048×512 × 2048×512 | 24 | 1.1e9 | 2.7e6 | 400 | 3.8% |
| q6_K | 2048×512 × 2048×512 | 8 | 1.1e9 | 3.0e6 | 363 | 1.4% |

bytes 组成：W = 每权重字节 × K×N（q4_K 144/256≈0.5625 B，q6_K 210/256≈0.820 B）；A = fp16 激活 ×2 B，读一遍。

**所有大 MUL_MAT 的 AI（算术强度）都在 360~1500，远超 ridge 58 → 理论上 deep compute-bound（深度计算受限）。**

## 当前算力利用率

- pp512 实测 **869 t/s**（无 profiling）→ 一次 graph 耗时 ≈ 512/869 ≈ 0.589 s
- 一次 graph FLOPs（形状求和）≈ **0.94 TFLOPs**（sanity：1.24B × 2 × 512 = 1.27e12，形状法不含 FLASH_ATTN 等，略低）
- 实测算力 ≈ **1.6~2.2 TFLOPS**
- FP16 峰值利用率 ≈ **28%~37%**

## 结论

1. **PP 下 MUL_MAT 是 compute-bound（计算受限）**：限制因素是 GPU 算力，不是内存带宽 → 优化 kernel（内核）计算效率有真实空间。
2. **当前 FP16 利用率只有 ~30-37%**，远低于理论峰值 → 与"MUL_MAT 是热点"吻合，值得深挖。
3. 乐观上限：若利用率提到 70%，理论提速 ~2.5×（实际不可能整图跑满，但方向明确）。

## 为什么 compute-bound（计算受限）但利用率只有 30%？（待验证的假设）

1. **dequant（反量化）开销**：q4_K 需要解量化，dequant 指令可能摊薄有效 FP16 算力 —— 这是 Q4_K 模型最常见的损耗点。
2. **kernel（内核）效率**：ggml-metal 的 MUL_MAT kernel 的 tile（分块）/ simdgroup（SIMD 组）配置是否匹配 M=512 的形状。
3. **访存模式**：AI（算术强度）高 ≠ 实际不 stall（停顿）；若 cache blocking（缓存分块）不佳，仍会有 memory stall（内存停顿）掩盖算力。

## 不确定性

- 5.8 TFLOPS 是理论峰值，实际 Metal 可达峰值更低 → 真实利用率可能比 30-37% 高。
- FLOPs 估算误差 ±20%（形状法偏低、粗算法偏高）。
- **结论方向（compute-bound，计算受限）稳健**；具体百分比待 Instruments 或更高精度方法验证。

## 下一步（衔接里程碑 C Experiment 001）

- 读 `ggml/src/ggml-metal.m` 的 MUL_MAT Q4_K kernel（内核）：是否用了 `simdgroup`（SIMD 组）矩阵指令、tile（分块）大小、dequant（反量化）流水。
- 若装 Xcode，用 Instruments 验证实际 GPU 算力利用率，收敛 30-37% 的不确定性。
- 由此确定 Experiment 001 的具体优化方向（dequant 流水 / tile 配置 / 指令选择）。

---

# Experiment 001：验证 dequant（反量化）是 PP 下 MUL_MAT 的主损耗（里程碑 C，2026-08-31）

## 0. Kernel（内核）静态分析（前置，读代码）

### 调度配置（M2 实际走的路径）

- PP512 的 MUL_MAT 走 `ggml_metal_op_mul_mat` 的 **mat-mm**（matrix-matrix，矩阵乘）分支（`ne11=batch=512 > 8`）。
- `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices` → M2（Apple8 family）**不支持 tensor API**，走 legacy kernel（`mul_mm.metal` 的 `#else` 分支，显式 `simdgroup`（SIMD 组）指令）。
- threadgroup（线程组）配置（`ggml-metal-device.cpp` 非 tensor 分支，宏见 `ggml-metal-impl.h`）：

| 项 | 值 | 来源 |
|---|---|---|
| 输出 tile（分块） | **64 × 32**（权重列 × batch） | `nr0=64, nr1=32` |
| simdgroup（SIMD 组）/threadgroup | **4** | `N_MM_SIMD_GROUP_X*Y = 2×2` |
| threadgroup 线程数 | 64 | 4 × SZ_SIMDGROUP(16) |
| smem（共享内存） | **6144 B** = 4096(sa) + 2048(sb) | 64×32×2B + 32×32×2B |
| K tile（内层 K 步长） | **32** | `NK = SZ_SIMDGROUP*N_MM_NK = 16×2` |
| 矩阵乘指令 | `simdgroup_float8x8`（8×8 半精度） | 每 simdgroup 8 个累加器 `mc[8]` |

### Kernel 执行结构（legacy 路径，每个 K=32 迭代）

1. **PHASE 1（加载 + dequant（反量化））**：64 线程，每线程 2 个 work item，各 dequant 一个 16 权重块（`dequantize_q4_K`），写进 sa（smem）；激活 B 读入 sb。
2. `threadgroup_barrier`（线程组屏障）。
3. **PHASE 2（矩阵乘）**：4 次 8×8 步进，`simdgroup_load` 从 smem 载入 `ma[4]/mb[2]`，8 次 `simdgroup_multiply_accumulate`（乘加）。
4. `threadgroup_barrier`。

### `dequantize_q4_K`（dequantize.h:431）实现

- 每个 block_q4_K = 256 权重（8 组 × 32）；`il∈[0,8)` 选择 16 权重子块。
- 从 device 读 8B `qs` + 2B `scales` + `d/dmin`，核心循环 16 次：`reg[i] = dl*(q[i]&mask) - ml`（掩码→乘→减）。
- 含 `il<2`、`il/4` 等三元分支与移位索引计算。

### 发现的低效点（→ Experiment 001 假设）

1. **dequant（反量化）与矩阵乘无重叠**：PHASE 1 → barrier → PHASE 2 → barrier 严格串行。dequant 期间矩阵乘单元闲置。
2. **K tile=32 过小、barrier（屏障）频繁**：K=2048 → 64 次迭代 × 每次 2 次 threadgroup_barrier + 4×3 次 simdgroup_barrier。
3. **smem 仅 6KB**：不足以双缓冲（double-buffer，需 12KB，Apple GPU 上限 32KB 内，可行），dequant 延迟无法隐藏。

## 1. 实验设计（零代码：q4_K vs F16）

用 `llama-quantize --allow-requantize` 把 Q4_K_M 模型反量化成 F16（2.3 GiB，16 BPW），`llama-bench` 同参数对比。F16 权重字节是 q4_K 的 **4.1 倍**（16 vs 5.18 BPW）——若 PP 仍 compute-bound（计算受限），带宽翻 4 倍影响小，差异即反映 dequant 计算开销。

```
./build/bin/llama-bench -m models/<模型> -p 512 -n 32 -r 2 -ngl -1
LLAMA_OP_PROFILE=1 ./build/bin/llama-bench ...（拿 MUL_MAT 绝对时间）
```

## 2. 结果

### llama-bench 吞吐（无 profiling）

| 模型 | 权重 BPW | pp512 | tg32 |
|---|---|---|---|
| Q4_K | 5.18 | **994.21 ± 1.27** | **93.74 ± 3.26** |
| F16 | 16.00 | **1208.40 ± 4.04**（+21.5%） | **36.44 ± 0.14**（−61%） |

### MUL_MAT 绝对时间（带 profiling，pp512）

| 模型 | MUL_MAT time | 占 graph 比 |
|---|---|---|
| Q4_K | 1028.8 ms | 87.26% |
| F16 | 834.7 ms | 84.13% |

**q4_K MUL_MAT 比 F16 慢 23.3%**（与 tok/s 差异 21.5% 一致）。

### Experiment 001b：q8_0 对照（dequant（反量化）复杂度 vs 访存字节）

用 `llama-quantize --allow-requantize` 再转一个 q8_0（dequant 极简：每 32 权重只 1 个 scale，无 mask/min/分支），与 q4_K/F16 组成三点对照。

| 模型 | dequant 复杂度 | 权重 BPW | pp512 | MUL_MAT 时间 | tg32 |
|---|---|---|---|---|---|
| q4_K | 复杂（mask + min + 分支） | 5.18 | 994.21 | 1028.8 ms | 93.74 |
| **q8_0** | **极简（单 scale）** | **8.50** | **1094.82**（+10%） | **923.7 ms**（−11%） | 56.88 |
| F16 | 无 | 16.00 | 1208.40（+21%） | 834.7 ms（−23%） | 36.44 |

**解读**：
- PP 下 q8_0 权重字节比 q4_K 多 **64%**，反而快 **10%** → 瓶颈不是访存字节量，而是 **dequant 的 ALU（算术逻辑单元）计算复杂度**。
- 三组单调：dequant 越简单 → PP 越快（994 < 1095 < 1208）；TG 严格反向（93.7 > 56.9 > 36.4，带宽墙佐证）。
- **结论：优化靶点 = dequantize_q4_K 的计算路径（位操作、分支、索引），而非 NK（内层 K 步长）/barrier（屏障）结构本身。** NK 翻倍/双缓冲是次要方向，仅在 dequant 已优化后才考虑。

### Experiment 001c：kernel 内探针（bypass dequant（反量化）计算，2026-08-31）

Metal kernel 运行时编译，改 `mul_mm.metal` legacy 分支第 232 行，把 `dequantize_func(x, il, temp_a)` 替换为「跳过 dequant 计算、保留 1 字节 device 读取（防编译器删读）、其余填常数」。测量 pp512 MUL_MAT 时间（`LLAMA_OP_PROFILE=1`，q4_K 模型）。

**结果**：

| 变体 | pp512 MUL_MAT | vs q4_K baseline |
|---|---|---|
| q4_K baseline | 1028.8 ms | — |
| **q4_K + bypass dequant** | **912.1 ms** | **−11.3%** |
| q8_0（极简 dequant） | 923.7 ms | −10.2% |
| F16（无 dequant） | 834.7 ms | −18.9% |

**成本分解（q4_K 1028.8 ms）**：

```
834.7  F16 基线（无 dequant、规则对齐访存）→ 对应 MMA 指令利用率 ~38%（上限）
 +92   dequant 计算成本（探针 → −11.3%）
 +102  4-bit 权重访存解包 / kernel 分支（探针 → F16 剩余差距）
1028.8 q4_K baseline
```

**颠覆性结论**：
1. **dequant（反量化）计算只占 MUL_MAT 的 ~11%，不是主瓶颈。** 双缓冲（隐藏 dequant）的理论上限就是这 ~11%，且实现复杂（smem（共享内存）布局与 NK 索引深度耦合），**性价比低，放弃作为首选优化**。
2. **真正的天花板是 MMA（矩阵乘）指令利用率本身：F16（零 dequant）也只有 38%。** 利用率 30-37% 的主因不是 dequant，而是 kernel 的 tile（分块）/simdgroup（SIMD 组）8×8 指令组合在 PP512 形状下达不到峰值。→ 优化方向应转向「提高 MMA 指令利用率」，而非 dequant。
3. 剩余 ~10%（探针→F16）来自 4-bit 权重访存解包与 kernel 分支——q4_K 格式固有，难改。
4. 探针（912ms）≈ q8_0（924ms）交叉印证了「dequant 计算 ≈ 4-bit 访存解包」各自 ~10% 的量级。

### Experiment 001d：PP batch 扫描（M=128→2048，利用率是否随 M 提升）

`llama-bench -p 128,256,1024,2048 -n 0 -r 2 -ngl -1`（无 profiling 干扰）：

| pp（=M/batch） | tok/s | 单次 MUL_MAT（profile 口径） |
|---|---|---|
| 128 | 964.5 ± 22.2 | 312 ms（calls=226） |
| 256 | 958.0 ± 47.1 | 534 ms（calls=226） |
| 512 | 994.2 | 1028.8 ms |
| 1024 | 975.8 ± 1.0 | 511 ms（calls=113） |
| 2048 | 923.5 ± 1.7 | 502 ms（calls=113） |

**结论**：M 从 128→2048，tok/s 稳定在 923-994，**利用率不随 M 提升** → 排除「batch 太小、threadgroup（线程组）不足」的假设。M=512 时利用率已到该 kernel 的水平上限。

## 3. 结论

1. **假设 1 确认：dequant（反量化）是 PP 下 MUL_MAT 的主损耗之一。** PP（compute-bound，计算受限）下，q4_K 权重字节只有 F16 的 1/4，MUL_MAT 反而慢 **23%** → 瓶颈不在权重带宽，在 q4_K kernel 的 dequant 计算路径。若把 dequant 开销消除/减半，PP 有 **~20% 提潜**。
2. **反向佐证带宽结论**：TG 下 F16 慢 61%（≈ 4.1× 字节 → ~3× 慢，接近 100 GB/s 带宽墙）→ 量化对 TG 至关重要，不能靠换 F16 提升，只能优化 q4_K kernel 本身。
3. **上一章"为什么利用率只有 30%"的 3 个候选中，dequant 被数据钉死为主因；kernel tile 配置与访存模式为次要。**

## 4. 下一步（Experiment 002 候选方向）

按投入/产出排序：
- **A. K tile 翻倍（NK 32→64）+ 双缓冲（double-buffer）smem**：barrier 减半 + dequant 与矩阵乘重叠。改 legacy kernel + device 层 smem 分配（6144→12KB）。
- **B. dequant 微优化**：`dequantize_q4_K` 用 `uchar4` 宽读、消除 `il` 分支的移位索引。
- **C. 用 profile 对比改动前后**：MUL_MAT 绝对时间是否下降（同一 LLAMA_OP_PROFILE 口径）。

---

# Experiment 002：纯 Metal GEMM 微基准——隔离单个 F16 MUL_MAT（里程碑 C，2026-09-02）

## 0. 本节要回答的问题

Experiment 001 把 F16 模型（无 dequant（反量化））的 MUL_MAT 基线压到 834.7 ms，对应利用率 ~38%（对 5.8 TFLOPS FP16 峰值）。**但 38% 是在整张 GGML 图里测的**：图里有 RMS_NORM、attention、多个 MUL_MAT 串行调度、每 op 一次 dispatch + 同步。

问题：这个 38% 是「ggml kernel 本身只有这么高」，还是「kernel 被嵌在整图里，被调度/同步/前后算子拖累，隔离跑能高得多」？

之前一直缺一个测量：**把单个 MUL_MAT 从图里摘出来，纯跑一个 kernel，看它能到多少。** 本实验补上它。

## 1. 实验设计（新增工具 examples/gemm-bench）

新写一个独立可执行 `llama-gemm-bench`（源码 [examples/gemm-bench/gemm-bench.cpp](../../examples/gemm-bench/gemm-bench.cpp)）：

- workload 压成一个纯 GEMM：权重 W(K×N, F16) × 激活 X(K×M, F32) → 输出 Y，只由 Metal backend 单独调度这一个 op；
- 形状对齐 Llama-3.2-1B 真实层（K=hidden=2048，N 为 2048/8192），再扫 M=512→4096，另加一个极限形状看是否饱和；
- 每个形状在 graph 里复制若干次摊销启动开销，warmup 3 次后取 7 次中位数；
- 峰值假设与全篇一致：M2 FP16 peak = 5.8 TFLOPS（环境变量 `PEAK_TFLOPS` 可覆盖）。

构建（已注册进 examples/CMakeLists.txt）：

```bash
cmake --build build --target llama-gemm-bench
```

运行（默认形状集；也可 `llama-gemm-bench K N M [label]` 跑自定义单个形状）：

```bash
./bin/llama-gemm-bench
```

> 注意：本机构建 `GGML_BACKEND_DL=ON`（动态加载后端），Metal 后端是 dlopen 进来的 MODULE 库，其内部全局 device 容器在进程正常 return 时会被二次释放而 abort，故工具在打印完结果后 `_exit(0)` 跳过库析构（不影响测量）。

## 2. 结果（原始输出）

```
|     K    |     N    |    M   | label                                        | GFLOP/disp |  rep |   us/disp | TFLOPS |  util |
|   2048 |   8192 |   512 | FFN up (K=2048 N=8192) x M=512   (模型 gate/up 层) |    17.18 |    1 |    7163.0 |  2.40 |  41.4% |
|   8192 |   2048 |   512 | FFN down (K=8192 N=2048) x M=512 (模型 down 层) |    17.18 |    1 |    7217.0 |  2.38 |  41.0% |
|   2048 |   2048 |   512 | attn proj (K=2048 N=2048) x M=512 (q/k/v/o)    |     4.29 |    5 |    1783.8 |  2.41 |  41.5% |
|   2048 |   8192 |  1024 | FFN up x M=1024 (batch 变大)                 |    34.36 |    1 |   13960.0 |  2.46 |  42.4% |
|   2048 |   8192 |  2048 | FFN up x M=2048 (batch 变大)                 |    68.72 |    1 |   27568.0 |  2.49 |  43.0% |
|   2048 |   8192 |  4096 | FFN up x M=4096 (batch 更大)                 |   137.44 |    1 |   54813.0 |  2.51 |  43.2% |
|   4096 |   8192 |  4096 | saturate: K=4096 N=8192 M=4096 (极限形状)    |   274.88 |    1 |  109991.0 |  2.50 |  43.1% |
```

## 3. 结论

1. **隔离单 kernel（模型 FFN 层形状）= 41.4%，整图 llama-bench F16 ≈ 38%** → 图调度/同步税只有 ~3 个百分点，不是 38% 的主因。kernel 被摘出来也上不了多少。
2. **M 从 512 一路扫到 4096（K 到 4096），利用率只从 41.4% 爬到 43.2% 后平台** → 不是「任务不够大 / occupancy（驻留线程数）不足」。即使喂一个单 dispatch 就 275 GFLOP 的极限 GEMM，kernel 依旧卡在 ~43%。这是指令级天花板，不是资源调度问题。
3. **43% 的墙与 dequant 无关**（这是纯 F16，没有反量化）。此前判断「真瓶颈是 MMA（矩阵乘指令）效率本身」现在有了直接实证：8×8 simdgroup（SIMD 组）矩阵指令的 issue（发射）效率 + NK=32 每次迭代 barrier（栅栏）串行结构，决定了每时钟能发出的有用 MMA 数量上限。
4. 因此优化锚点更清晰：dequant（116.7 ms）+ 4-bit 解包（77.4 ms）叠加在这面 43% 的墙**之上**；真正能拉开差距的是先提高墙本身（NK 翻倍 + 双缓冲，barrier 减半 + 使 dequant/load 与 MMA 重叠），而不是在 q4_K 里抠分支。

## 4. 诚实声明（这个实验的边界）

- **绝对利用率依赖 5.8 TFLOPS 的 peak 假设**。若真实 FP16 peak 接近 Apple 公布的 FP32 值（≈3.6 TF），绝对 util 会按比例更高——但**「隔离 ≈ 整图」「M 扫不上去」这两个相对结论不依赖该假设**，依然成立。
- 单 kernel 上限 ~43% 是对 **这个 legacy kernel（M2 无 tensor core 路径）** 而言；不代表 M2 硬件只能做到 43%（更优的 tiling / 调度理论上可更高）。它代表的是 **ggml 当前 kernel 的天花板**——这正是本系列要量化的对象。

## 5. 下一步

Experiment 002 给了 A/B 方案的量化收益边界：NK=64 + 双缓冲的预期收益是「整图 38% → 靠近 43% 的 kernel 墙」之间的差（约 5 pp），而不是 38% → 100%。是否值得做、做完能否叠上 dequant 优化，需按此边界重新评估。

---

# Experiment 003：43% 墙的归因实验（2026-09-02）

## 0. 本节要回答的问题

Experiment 002 把 38%（整图）和 43%（隔离单 kernel）之间的墙锁定了：**无论 M 多大，F16 kernel 最多 ~43%**。现在的问题是归因——这面墙是「kernel 编排浪费」（同步、指令密度），还是「Apple 8×8 MMA（矩阵乘指令）本身的吞吐上限」？

拆成可分别证伪的假设，用 `llama-gemm-bench --verify`（Metal vs CPU 数值校验）兜底正确性：

- H1：PHASE2 每迭代的 12 个 `simdgroup_barrier`（SIMD 组同步）+ 每 K-tile 的 2 个全组 `threadgroup_barrier` 占用发射/等待 → 墙的一部分是同步。
- H2：每个 K-tile 的固定开销（PHASE1 装货 + 边界）→ NK 翻倍能摊薄。
- H3：PHASE2 的 smem→寄存器 load 指令密度（每 8 层 K 需 4+2 条 load 喂 8 条 MMA）是墙 → 任何不改变 load:MMA 比例的改动都无效。

## 1. Experiment 003-S1：删除 PHASE2 的 mem_none barrier（2026-09-02）

### 改动

[kernels/mul_mm.metal](../../ggml/src/ggml-metal/kernels/mul_mm.metal) legacy kernel（145 行起）PHASE2 里每个 ik 迭代的 3 处 `simdgroup_barrier(mem_flags::mem_none)`（原 294/300/306 行）删除，仅作用于 `kernel_mul_mm`（`kernel_mul_mm_id` 不动）。理由：上一行 `threadgroup_barrier`（行 287）已保证 smem 写对全部 simdgroup 可见；此后只读 smem、写寄存器，同一 simdgroup 内 load→MMA 的顺序由寄存器数据依赖保证，这些 barrier 理论上是冗余。

```bash
cp ggml/src/ggml-metal/kernels/mul_mm.metal /tmp/mul_mm.metal.bak_exp_s1
# 按行号删三处 barrier,插入实验注释
cmake --build build --target llama-gemm-bench   # GGML_METAL_EMBED_LIBRARY=ON,改 .metal 后重编即可生效
./bin/llama-gemm-bench --verify                 # 数值校验必须先 PASS
```

### 结果

| 形状 | baseline | S1 | 变化 |
|---|---|---|---|
| FFN up (2048×8192×512) | 41.4% | 40.9% | −0.5pp（噪声内） |
| attn proj (2048×2048×512) | 41.5% | 40.8% | −0.7pp（噪声内） |
| saturate (4096×8192×4096) | 43.1% | 42.6% | −0.5pp（噪声内） |

`--verify`: rel=0, PASS。

### 结论

**H1 证伪：同步指令 ≈ 免费。** 删掉每 K-tile 12 个 barrier 对吞吐没有任何可测量的正收益。量化估算也支持这一点：每个 K-tile 的 barrier 即使按 ~200 cycle 计，占整个 threadgroup 生命周期（K=2048 时 ~9.4M cycle @2.40 TFLOPS）不到 1%。墙不在这。

推论：**H2（NK 翻倍摊薄固定开销）的预期收益也随之趋近于 0**——它的主要候选收益来源（barrier/装货固定开销）已被 S1 证明不占关键路径。真正剩余的主因是 H3：PHASE2 里 load 指令与 8×8 MMA 竞争指令吞吐（每推进 8 层 K 需要 6 条 load 喂 8 条 MMA，算术指令密度 ~46%，与观测 ~43% 同量级）。

### 下一步候选（据 S1 重新排序）

- A. **合成 MMA 探针**（最有分辨力）：写一个数据全在寄存器、只循环发 `simdgroup_multiply_accumulate` 的最小 Metal kernel，测 Apple 8×8 fp16 指令的**硬件可达吞吐**。若纯 MMA 也只有 ~50-60%，43% 就是硬件给的；若接近 100%，则 ggml 的 load 结构确实浪费了 ~57%。这个实验独立于 ggml 调度，能一次性回答"墙是硬件还是 kernel"。
- B. 仍做 NK64（预期收益≈0，仅用于穷尽 H2 的最后一个变体）。

## 2. Experiment 003-S2：合成 MMA 探针（2026-09-02）

### 改动

新增独立工具 [examples/mma-probe/mma-probe.mm](../../examples/mma-probe/mma-probe.mm)（target `llama-mma-probe`，只依赖系统 Metal.framework，不链接 ggml）。kernel 把一对 8×8 fp16 矩阵 A、B 填进寄存器，循环 iters 轮，每轮发 8 条**互相独立**的 `simdgroup_multiply_accumulate`（累加器 C0..C7），全程没有 smem、没有 load、没有 barrier。结果只 `simdgroup_store` 一次防止死代码消除。内置线性度自检（iters 与 2×iters 时间比应≈2）。

```bash
cmake --build build --target llama-mma-probe
./bin/llama-mma-probe 32768 200000
```

### 结果

| 变体 | threads | ILP（独立累加器） | TFLOPS | util%（对 5.8） | 线性度 |
|---|---|---|---|---|---|
| 小计算量（无效，launch 主导） | 8192 | 4 | 0.81 | 13.9% | 0.93（作废） |
| 大计算量 | 32768 | 4 | 2.71 | 46.7% | 1.98 ✓ |
| 更多线程（occupancy↑） | 65536 | 4 | 2.68 | 46.1% | 1.99 ✓ |
| ILP 翻倍 | 32768 | 8 | 2.78 | 48.0% | 2.00 ✓ |

### 结论

**纯 8×8 fp16 MMA 指令流（无任何访存/同步/解量化）的硬件上限 ≈ 2.7-2.8 TFLOPS，即对 5.8 TF 假设的 ~48%。** 三个变体全部封顶在 46-48%：加线程（occupancy）不涨、ILP 4→8 只涨 1.3pp——排除"探针自己没写满"的可能。这是指令吞吐级的天花板。

对照 ggml F16 kernel（含全部 load/smem/barrier）：41-43% ≈ 2.4-2.5 TF。

- **ggml legacy kernel 已到该指令路径硬件上限的 ~88%**（2.45 / 2.78）。43% 与纯 MMA 48% 之间仅剩 5pp 是 ggml 的访存/编排开销——而它（被 Experiment 001-003-S1 排除 dequant、barrier 后）主要是 PHASE2 的 load 指令，但那是 8×8 指令模型的硬约束，几乎不可再削。
- **5.8 TFLOPS 假设对"8×8 simdgroup fp16 指令"路径不成立**：纯 MMA 最佳也只有 ~2.8 TF，说明 M2 这条路径的实际可达到峰值远低于 5.8，更接近 Apple 公布的 FP32 数值（8 核 ≈3.6 TF，Apple GPU 的 fp16 不提供双倍吞吐）。若以 3.6 TF 计，ggml kernel 实际已达到硬件 ~68%，纯 MMA ~77%。
- **最终归因**：PP 下 MUL_MAT 的 ~38%（整图）主要不是"kernel 写得低效"，而是 M2 无 tensor core（`has_tensor=false`），8×8 simdgroup fp16 指令本身只能提供 ~2.8 TF 的吞吐。要实质性突破只能换机器（带 tensor 的 GPU 家族）或换精度/权重格式，而不是改这个 legacy kernel 的编排。

> 一句话：43% ≈ 48% 的硬件墙（纯 MMA）+ ~5pp 的访存编排税。dequant 不是瓶颈（001）、同步不是瓶颈（S1）、load 编排只占 ~5pp（S2 归因完毕）——**结论：M2 legacy kernel 已接近它所在指令路径的物理上限，PP kernel 优化在此硬件上无大空间。**



