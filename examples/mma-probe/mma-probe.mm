// llama-mma-probe:合成 MMA(矩阵乘指令)探针
//
// 目的:直接测 Apple 8×8 fp16 simdgroup_multiply_accumulate(SIMD 组乘累加)
// 指令的硬件可达吞吐,不经过 ggml/llama 的任何调度与访存结构。
//
// 为什么需要它(承接 docs/performance/analysis.md Experiment 002/003):
//   llama-gemm-bench 测出纯 F16 MUL_MAT kernel 上限 ~43%(对 5.8 TFLOPS 假设)。
//   Experiment 003-S1 已证明同步(barrier)几乎免费,墙被指向"load 指令密度"。
//   这个探针把 load 全部拿掉——数据全在寄存器、循环里只有 MMA——给出一个参照系:
//     - 若纯 MMA 也只有 ~50%,说明 43% 基本是 Apple 8×8 指令的硬件上限,ggml 已接近它;
//     - 若纯 MMA 接近 100%,说明 ggml 的 smem→寄存器 load 结构确实浪费了约一半。
//
// kernel 逻辑:
//   每个 SIMD 组持有一对 8×8 fp16 矩阵 A、B 和 4 个 8×8 f32 累加器 C0..C3,
//   循环 iters 次,每轮发 4 条互相独立的 MMA(C_i += A*B)。数据永不从内存读,
//   只把最后的 C 通过 simdgroup_store 写回一次,防止死代码消除。
//   (C_i 是跨轮累积的串行依赖链,编译器无法把"结果没被读"的循环折叠掉;同时
//    4 条独立链提供指令级并行。内置线性度自检:iters 与 2×iters 的时间应约等于 2 倍,
//    若远小于 2 倍说明循环被编译器优化掉了,结果作废。)
//
// 用法:
//   llama-mma-probe [threads] [iters]
//
// 输出:
//   device | threads | simdgroups | iters | ms | TFLOPS | util%(默认对 5.8 TFLOPS,
//   环境变量 PEAK_TFLOPS 可覆盖) + 线性度自检两行

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char * kMSL = R"(
using namespace metal;

kernel void mma_burn(
        device float * out [[buffer(0)]],
        constant uint & iters [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
{
    // 8x8 fp16 乘数,全在寄存器(探针关键:没有 smem load,没有 barrier)
    simdgroup_half8x8  A = make_filled_simdgroup_matrix<half, 8>(half(0.5));
    simdgroup_half8x8  B = make_filled_simdgroup_matrix<half, 8>(half(0.25));

    // 8 个 f32 累加器:8 条独立依赖链,提高指令级并行(验证 ILP 是否够)
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C4 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C5 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C6 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 C7 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(C0, A, B, C0);
        simdgroup_multiply_accumulate(C1, A, B, C1);
        simdgroup_multiply_accumulate(C2, A, B, C2);
        simdgroup_multiply_accumulate(C3, A, B, C3);
        simdgroup_multiply_accumulate(C4, A, B, C4);
        simdgroup_multiply_accumulate(C5, A, B, C5);
        simdgroup_multiply_accumulate(C6, A, B, C6);
        simdgroup_multiply_accumulate(C7, A, B, C7);
    }

    // 结果落地,防止编译器把上面的循环当"死代码"整体删掉。
    // 每个 SIMD 组写自己的一块区域(8 块 × 8 行 × 8 列 f32),互不冲突即可。
    const uint sg = gid / 32;                       // 全局 SIMD 组号
    device float * dst = out + sg * 512;            // 每 SIMD 组 8×64 = 512 f32
    simdgroup_store(C0, dst,      8, 0, false);
    simdgroup_store(C1, dst +  64, 8, 0, false);
    simdgroup_store(C2, dst + 128, 8, 0, false);
    simdgroup_store(C3, dst + 192, 8, 0, false);
    simdgroup_store(C4, dst + 256, 8, 0, false);
    simdgroup_store(C5, dst + 320, 8, 0, false);
    simdgroup_store(C6, dst + 384, 8, 0, false);
    simdgroup_store(C7, dst + 448, 8, 0, false);
}
)";

// 跑一次 dispatch,返回毫秒
static double run_once(id<MTLComputePipelineState> ps, id<MTLCommandQueue> q,
                       id<MTLBuffer> outBuf, id<MTLBuffer> itersBuf, uint threads) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:outBuf   offset:0 atIndex:0];
    [enc setBuffer:itersBuf offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    const auto t0 = std::chrono::steady_clock::now();
    [cb commit];
    [cb waitUntilCompleted];
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double peak_tflops() {
    const char * env = std::getenv("PEAK_TFLOPS");
    return env ? atof(env) : 5.8;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        uint threads = (argc > 1) ? (uint) atoi(argv[1]) : 8192;
        uint iters   = (argc > 2) ? (uint) atoi(argv[2]) : 2048;

        NSError * err = nil;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "没有 Metal 设备\n");
            return 1;
        }

        NSString * src = [NSString stringWithCString:kMSL encoding:NSUTF8StringEncoding];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "MSL 编译失败: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mma_burn"];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) {
            fprintf(stderr, "pipeline 创建失败: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLCommandQueue> q = [dev newCommandQueue];

        // out buffer:每 SIMD 组 8 块 × 64 f32;iters buffer:1 个 uint
        const uint n_sg = threads / 32;
        id<MTLBuffer> outBuf  = [dev newBufferWithLength:n_sg * 64 * 8 * 4
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> itersBuf = [dev newBufferWithLength:4
                                     options:MTLResourceStorageModeShared];

        const uint n_iters = iters;
        auto run = [&](uint it) -> double {
            memcpy(itersBuf.contents, &it, 4);
            // warmup(驱动编译)
            run_once(ps, q, outBuf, itersBuf, threads);
            // 计时取 5 次中位数
            std::vector<double> samples;
            for (int r = 0; r < 5; ++r) {
                samples.push_back(run_once(ps, q, outBuf, itersBuf, threads));
            }
            std::nth_element(samples.begin(), samples.begin() + 2, samples.end());
            return samples[2];
        };

        const double t1 = run(n_iters);
        const double t2 = run(n_iters * 2);

        // 每秒 FLOP = SIMD 组数 × 每轮 8 条 MMA × 每条 8×8×8×2 FLOP
        const double flops_per_sg_iter = 8.0 * 8 * 8 * 8 * 2;  // 8192
        auto report = [&](double ms, uint it) {
            const double tf   = (double) n_sg * flops_per_sg_iter * (double) it / (ms * 1e-3) / 1e12;
            const double util = tf / peak_tflops() * 100.0;
            fprintf(stderr, "# iters=%-7u ms=%8.3f  TFLOPS=%6.2f  util=%5.1f%%\n", it, ms, tf, util);
        };
        report(t1, n_iters);
        report(t2, n_iters * 2);

        printf("# device=%s | threads=%u | simdgroups=%u | peak 假设=%.1f TFLOPS\n",
               dev.name.UTF8String, threads, n_sg, peak_tflops());
        printf("# 线性度自检:iters=%u→%.3f ms,2×iters=%u→%.3f ms,比值=%.2f(应≈2;若明显小于 2,\n"
               "#            说明循环被编译器优化掉,本次结果作废)\n",
               n_iters, t1, n_iters * 2, t2, t2 / t1);
    }
    return 0;
}
