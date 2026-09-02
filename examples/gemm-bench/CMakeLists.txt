# 纯 Metal GEMM 微基准:测单个 F16 MUL_MAT dispatch 的可达吞吐
set(TARGET llama-gemm-bench)

add_executable(${TARGET} gemm-bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
