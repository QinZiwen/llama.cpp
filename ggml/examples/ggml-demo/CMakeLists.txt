set(TARGET ggml_demo)
add_executable(${TARGET} ggml-demo.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)