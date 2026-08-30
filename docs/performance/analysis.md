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

