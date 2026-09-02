# 合成 MMA 探针:在纯 Metal 里测 8x8 fp16 simdgroup 矩阵乘指令的硬件可达吞吐。
# 只依赖系统 Metal.framework,不链接 llama/ggml。
if(NOT APPLE)
    message(WARNING "llama-mma-probe 只在 macOS(有 Metal)上可用,跳过")
    return()
endif()

set(TARGET llama-mma-probe)

add_executable(${TARGET} mma-probe.mm)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PRIVATE -fobjc-arc)

find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
target_link_libraries(${TARGET} PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
