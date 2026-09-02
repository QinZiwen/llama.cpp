// llama-gemm-bench:纯 Metal GEMM 微基准
//
// 用途:
//   在隔离条件下测量单个 F16 权重 MUL_MAT dispatch 的可达吞吐,
//   与 llama-bench 整图(PP)测到的利用率做对比,判断 ggml 的 Metal
//   kernel 离这块硬件的真实上限还有多远。
//
//   为什么用它:
//   - llama-bench 里 MUL_MAT 是嵌在整张 GGML 计算图里跑的,同时还有
//     调度开销、同步开销、前后算子。无法只看"一个 kernel 本身能跑多快"。
//   - 这里把 workload 压成一个纯 GEMM:W(KxN,F16) x X(KxM) -> Y(NxM),
//     由 Metal backend 单独调度,时间 = 纯 kernel 时间。
//
// 用法:
//   llama-gemm-bench                    # 跑内置形状集(对齐 Llama-3.2-1B 层形状)
//   llama-gemm-bench K N M [label]      # 跑自定义单个形状
//   llama-gemm-bench --verify           # 数值校验:同一 GEMM 在 Metal 与 CPU 各跑一次
//                                       # 比较输出(改动 kernel 后必须过这里再谈性能)
//
// 输出列:
//   K x N x M    权重/激活内维与批次
//   GFLOP/disp   单次 dispatch 的计算量 = 2*K*N*M
//   rep          graph 内重复该 GEMM 的次数(摊销启动开销)
//   us/disp      单次 dispatch 的中位数耗时(微秒)
//   TFLOPS       实际吞吐
//   util%        吞吐 / M2 FP16 峰值(默认 5.8 TFLOPS,可用环境变量 PEAK_TFLOPS 覆盖)

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

// M2 (8-core GPU) FP16 峰值。假设值来自 Apple 规格(与性能分析文档一致)。
// 通过 PEAK_TFLOPS 环境变量可覆盖。
static double peak_tflops() {
    const char * env = std::getenv("PEAK_TFLOPS");
    return env ? atof(env) : 5.8;
}

// 单个 MUL_MAT 的几何:权重 W (K x N, F16),激活 X (K x M, F32)。
struct shape {
    int    K;        // 内维(reduction 维)
    int    N;        // 输出行数(输出 token 维,每权重列一个)
    int    M;        // 输出列数(batch / 序列长度)
    const char * label;
};

// 对齐 Llama-3.2-1B 的层形状(K 是模型 hidden=2048,n_ff=8192):
static const shape default_shapes[] = {
    { 2048,  8192,   512, "FFN up (K=2048 N=8192) x M=512   (模型 gate/up 层)" },
    { 8192,  2048,   512, "FFN down (K=8192 N=2048) x M=512 (模型 down 层)"   },
    { 2048,  2048,   512, "attn proj (K=2048 N=2048) x M=512 (q/k/v/o)"       },
    { 2048,  8192,  1024, "FFN up x M=1024 (batch 变大)"                       },
    { 2048,  8192,  2048, "FFN up x M=2048 (batch 变大)"                       },
    { 2048,  8192,  4096, "FFN up x M=4096 (batch 更大)"                       },
    { 4096,  8192,  4096, "saturate: K=4096 N=8192 M=4096 (极限形状)"           },
};

static constexpr int    N_GRAPH  = 1024;  // cgraph 容量,决定最多能重复多少次 dispatch
static constexpr int    REPS     = 7;     // 每个形状采样的次数(取中位数)
static constexpr int    WARMUPS  = 3;     // 预热次数(编译 Metal pipeline + 提频)

// 单次 dispatch 的耗时受启动开销污染,把同一个 GEMM 在 graph 里重复若干次,
// 让总时间落在 ~毫秒级,再用总时间除以次数(与 tools/tuning 的做法一致)。
static int n_runs_for(int64_t flops_per_dispatch) {
    const int64_t target_flops = 25LL * 1000 * 1000 * 1000;  // 每个 rep 想要 ~25 GFLOP
    const int     cap          = N_GRAPH - 1;                  // graph 里还能塞多少节点
    if (flops_per_dispatch <= 0) {
        return 1;
    }
    return std::max(1, std::min<int>(cap, (int) (target_flops / flops_per_dispatch)));
}

// 跑一个形状,返回中位数总耗时(us),并输出单次 dispatch 的 us 到参数
static double time_dispatch(ggml_backend_t backend, int K, int N, int M, int & n_runs) {
    const int64_t flops_dispatch = 2LL * K * N * M;

    ggml_init_params params = {
        /* mem_size = */ ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(N_GRAPH, false),
        /* mem_base = */ nullptr,
        /* no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx && "ggml_init failed");

    // W: K 内维(N x K 布局:ne0=K, ne1=N);X: K 内维(ne0=K, ne1=M)
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);  // Y: (N, M), F32

    // 张量交给 Metal backend 分配(no_alloc 模式)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf && "backend alloc failed");
    (void) buf;

    // 填充非零数据(不影响耗时,只保证结果有效可校验)
    {
        std::vector<ggml_fp16_t> w_data((size_t) K * N);
        std::vector<float>       x_data((size_t) K * M);
        for (size_t i = 0; i < w_data.size(); ++i) {
            w_data[i] = ggml_fp32_to_fp16((float) ((i & 3) + 1) * 0.25f);
        }
        for (size_t i = 0; i < x_data.size(); ++i) {
            x_data[i] = (float) ((i & 7) + 1) * 0.125f;
        }
        ggml_backend_tensor_set(W, w_data.data(), 0, w_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(X, x_data.data(), 0, x_data.size() * sizeof(float));
    }

    // 构建计算图
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, N_GRAPH, false);
    ggml_build_forward_expand(gf, Y);

    // 复制同一个 GEMM 节点,摊销启动开销
    n_runs = n_runs_for(flops_dispatch);
    for (int i = 1; i < n_runs; ++i) {
        ggml_graph_add_node(gf, Y);
    }

    // 预热
    for (int i = 0; i < WARMUPS; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // 采样取中位数
    std::vector<int64_t> samples;
    samples.reserve(REPS);
    for (int r = 0; r < REPS; ++r) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        samples.push_back(ggml_time_us() - t0);
    }
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());

    // 校验结果非零/有限(防止整图被调度器当空跑优化)
    {
        float y0 = 0.0f;
        ggml_backend_tensor_get(Y, &y0, 0, sizeof(y0));
        if (!(std::isfinite(y0) && y0 != 0.0f)) {
            fprintf(stderr, "# 警告:输出校验失败 y0=%f (K=%d N=%d M=%d)\n", y0, K, N, M);
        }
    }

    ggml_free(ctx);
    return (double) samples[samples.size() / 2];
}

// 数值校验:同一个 GEMM(Metal vs CPU)输出是否一致。
// 两个 backend 各自独立建图、分配、填同一份 host 数据,分别计算后逐元素比较。
// CPU backend 是数值参照——它走完全不同的内核路径,若 Metal kernel 被改错(如误删
// 必要的同步),输出的偏差会远大于两个合法实现之间正常的舍入差。
static bool verify_mul_mat(ggml_backend_t metal, int K, int N, int M) {
    std::vector<ggml_fp16_t> w_data((size_t) K * N);
    std::vector<float>       x_data((size_t) K * M);
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = ggml_fp32_to_fp16((float) ((i & 3) + 1) * 0.25f);
    }
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = (float) ((i & 7) + 1) * 0.125f;
    }

    // 在指定 backend 上:建图 → 填数据 → 计算 → 读回完整输出
    auto compute_on = [&](ggml_backend_t be, std::vector<float> & y_out) {
        ggml_init_params params = {
            ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx && "ggml_init failed");
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        GGML_ASSERT(buf && "alloc failed");
        ggml_backend_tensor_set(W, w_data.data(), 0, w_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(X, x_data.data(), 0, x_data.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(gf, Y);
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        y_out.resize((size_t) N * M);
        ggml_backend_tensor_get(Y, y_out.data(), 0, y_out.size() * sizeof(float));
        ggml_free(ctx);
    };

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "# verify:CPU backend 初始化失败,无法校验\n");
        return false;
    }

    std::vector<float> y_metal, y_cpu;
    compute_on(metal, y_metal);
    compute_on(cpu, y_cpu);
    ggml_backend_free(cpu);

    double max_abs_diff = 0.0;
    double max_abs      = 0.0;
    for (size_t i = 0; i < y_metal.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, (double) std::fabs(y_metal[i] - y_cpu[i]));
        max_abs      = std::max(max_abs, (double) std::fabs(y_cpu[i]));
    }
    const double rel = (max_abs > 0) ? max_abs_diff / max_abs : 0.0;
    const bool   ok  = rel < 1e-2;  // 两个合法实现之间的正常舍入差远小于 1%;若 kernel 被改错,rel 会到 O(1)
    fprintf(stderr, "# verify: Metal vs CPU (K=%d N=%d M=%d)  %zu elems | max_abs_diff=%.3e | max_abs=%.3e | rel=%.3e | %s\n",
            K, N, M, y_metal.size(), max_abs_diff, max_abs, rel, ok ? "PASS" : "FAIL");
    return ok;
}

// 打印一行结果
static void report(int K, int N, int M, const char * label, double us_per_dispatch, int n_runs) {
    const double flops = 2.0 * (double) K * (double) N * (double) M;
    const double tf    = flops / (us_per_dispatch * 1e-6) / 1e12;  // TFLOPS
    const double peak  = peak_tflops();
    const double util  = tf / peak * 100.0;

    printf("| %6d | %6d | %5d | %-46s | %8.2f | %4d | %9.1f | %5.2f | %5.1f%% |\n",
           K, N, M, label, flops / 1e9, n_runs, us_per_dispatch, tf, util);
}

int main(int argc, char ** argv) {
    ggml_time_init();

    // 解析参数:--verify 开关 + 可选的自定义形状 K N M [label]
    bool do_verify = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
        }
    }

    std::vector<shape> shapes;
    if (argc >= 4 && argv[1][0] != '-') {  // 自定义形状(不以 - 开头)
        shape s;
        s.K = atoi(argv[1]);
        s.N = atoi(argv[2]);
        s.M = atoi(argv[3]);
        s.label = (argc >= 5 && argv[4][0] != '-') ? argv[4] : "custom";
        shapes.push_back(s);
    } else {
        for (const auto & s : default_shapes) {
            shapes.push_back(s);
        }
    }

    // 初始化 Metal backend。
    // 本构建 GGML_BACKEND_DL=ON,Metal 是运行时从 libggml-metal.so 动态加载的,
    // 所以先 load_all() 再从注册设备里找 Metal(不能直接链接 Metal 符号)。
    // 注意:Metal 设备在这个版本里按"索引+序号"注册,名字形如 "MTL0"。
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "MTL", 3) == 0) {
            dev = d;
            break;
        }
    }
    ggml_backend_t backend = dev ? ggml_backend_dev_init(dev, NULL) : nullptr;
    if (!backend) {
        fprintf(stderr, "Metal backend 初始化失败。已注册的设备:");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        fprintf(stderr, "\n");
        return 1;
    }

    fprintf(stderr, "# backend: %s | FP16 peak 假设: %.1f TFLOPS (环境变量 PEAK_TFLOPS 可覆盖)\n",
            ggml_backend_name(backend), peak_tflops());

    // 数值校验(Metal vs CPU):改过 kernel 后必须先过这里
    if (do_verify) {
        if (!verify_mul_mat(backend, 512, 512, 512)) {
            fprintf(stderr, "# verify: FAILED —— Metal kernel 输出与 CPU 不一致,停止,不要测性能\n");
            _exit(1);
        }
        if (shapes.size() == 1 && strcmp(shapes[0].label, "custom") == 0 && argc == 2) {
            _exit(0);  // 只校验,不测性能
        }
    }

    // 表头
    printf("\n");
    printf("|     K    |     N    |    M   | %-46s | %8s | %4s | %9s | %6s | %5s |\n",
           "label", "GFLOP/disp", "rep", "us/disp", "TFLOPS", "util");
    printf("|----------|---------|--------|-%s-|----------|------|-----------|--------|--------|\n",
           std::string(46, '-').c_str());

    for (const auto & s : shapes) {
        int n_runs = 1;
        const double us_total = time_dispatch(backend, s.K, s.N, s.M, n_runs);
        report(s.K, s.N, s.M, s.label, us_total / n_runs, n_runs);
    }

    printf("|----------|---------|--------|-%s-|----------|------|-----------|--------|--------|\n",
           std::string(46, '-').c_str());
    printf("\n");

    // 对照提示(结论依据见 docs/performance/)
    fprintf(stderr,
            "# 对照:llama-bench 整图(PP512, F16 权重模型)在 M2 上测到约 38%% 的利用率。\n"
            "# 上面 util 列是本微基准在隔离条件下单个 kernel 能到多少,二者之差即\"图调度/同步税\"。\n");

    ggml_backend_free(backend);

    // 在 GGML_BACKEND_DL=ON 下,Metal 后端是 dlopen 进来的 MODULE 库,
    // 它内部的全局 device 容器在进程正常 return(经 __cxa_finalize 触发 C++ 静态析构)
    // 时会被再次释放导致 abort。基准结果此时已打印完毕,_exit 直接终止、跳过库析构。
    fflush(stdout);
    _exit(0);
}
