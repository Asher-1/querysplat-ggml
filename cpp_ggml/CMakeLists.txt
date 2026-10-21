cmake_minimum_required(VERSION 3.14)
project(querysplat_ggml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Embedded map-anything-ggml (pinned submodule) -------------------------
#
# Provides: the mapggml static library (gguf loader, backend wiring, graph
# builders incl. vggt_omega with RuntimeOptions::expose_tokens), the patched
# ggml submodule, and the upstream vggt-cli parity binary. GPU backends are
# requested the same way as upstream (-DMAP_GGML_CUDA=ON / VULKAN / METAL /
# HIPBLAS; at most one per build directory) and are forwarded to ggml by the
# subproject's CMakeLists.
set(MAP_GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/map-anything-ggml/cpp_ggml)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---- querysplat_core -------------------------------------------------------
# Everything querysplat-cli (inference) and querysplat-eval (the DL3DV
# evaluate_jsons.py protocol) share: image input, camera math, the decoder
# graph, host activations, rasterizer, python-format writers, metrics.
add_library(querysplat_core STATIC
    src/camera_math.cpp
    src/activations_host.cpp
    src/ply_writer.cpp
    src/output_parity.cpp
    src/dist_plot.cpp
    src/pyio.cpp
    src/image_io.cpp
    src/metrics.cpp
    src/lpips_graph.cpp
    src/querysplat_graph.cpp
    src/tto_loss.cpp
    src/render/rasterizer.cpp)
target_link_libraries(querysplat_core PUBLIC mapggml)
target_include_directories(querysplat_core PUBLIC src
    PRIVATE third_party/map-anything-ggml/cpp_ggml/third_party/stb)

# JPEG decode through the system libjpeg: PIL (the official chain) decodes
# JPEGs with libjpeg-turbo, whose ISLOW IDCT is bit-compatible with libjpeg's
# — stb_image's IDCT differs by +-1-2/255 per channel, which the model
# amplifies (courtyard cpp-vs-py was 30 dB for exactly this reason). Optional:
# without it, JPEG inputs fall back to stb (PNG/BMP/TGA unaffected).
find_package(JPEG QUIET)
if(JPEG_FOUND)
    target_compile_definitions(querysplat_core PRIVATE QS_USE_LIBJPEG)
    target_link_libraries(querysplat_core PUBLIC ${JPEG_LIBRARIES})
    target_include_directories(querysplat_core PRIVATE ${JPEG_INCLUDE_DIRS})
endif()

# WebP decode through the system libwebp (PIL's own WebP decoder library):
# infer.py accepts .webp inputs and stb_image has no WebP support. Optional:
# without it, .webp inputs fail to decode with a clear error.
find_path(WEBP_INCLUDE_DIR webp/decode.h)
find_library(WEBP_LIBRARY webp)
if(WEBP_INCLUDE_DIR AND WEBP_LIBRARY)
    target_compile_definitions(querysplat_core PRIVATE QS_USE_LIBWEBP)
    target_link_libraries(querysplat_core PUBLIC ${WEBP_LIBRARY})
    target_include_directories(querysplat_core PRIVATE ${WEBP_INCLUDE_DIR})
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(querysplat_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(querysplat_core PRIVATE QS_USE_OPENMP)
endif()

# ---- querysplat-cli --------------------------------------------------------
# Single-process dual-graph inference: the embedded mapggml runtime runs the
# VGGT-Omega encoder (expose_tokens) and this project's own ggml graph runs
# the QuerySplat decoder; host-side activation heads; forward CPU rasterizer.
add_executable(querysplat-cli src/cli.cpp)
target_link_libraries(querysplat-cli PRIVATE querysplat_core)

# ---- querysplat-eval -------------------------------------------------------
# C++ port of evaluations/evaluate_jsons.py (DL3DV input/interp/extrap
# protocol: target-camera prediction, Sim(3) alignment, 256x256 PSNR/SSIM/
# LPIPS-vgg) on the same runtime.
add_executable(querysplat-eval src/eval.cpp)
target_link_libraries(querysplat-eval PRIVATE querysplat_core)

# ---- querysplat_capi -------------------------------------------------------
# AICore-style C API over the dual-graph chain (mirrors mapggml/capi.h:
# opaque ctx + options setters + borrowed typed results, no env vars). The
# integration surface ACloudViewer binds against.
add_library(querysplat_capi SHARED src/querysplat_capi.cpp)
target_link_libraries(querysplat_capi PRIVATE querysplat_core)
target_include_directories(querysplat_capi PUBLIC include)
set_target_properties(querysplat_capi PROPERTIES DEFINE_SYMBOL QS_CAPI_BUILD)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(t querysplat_core querysplat-cli querysplat-eval querysplat_capi)
        target_compile_options(${t} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endforeach()
    # skimage SSIM is a chain of separately rounded numpy f32 ops: no FMA
    # contraction, or the port drifts off the reference by an ulp per op
    set_source_files_properties(src/metrics.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# ---- smoke tests (need models/gguf/*.gguf at runtime) ----------------------
option(QS_BUILD_TESTS "Build the smoke tests" ON)
if(QS_BUILD_TESTS)
    add_executable(test_smoke_tokens tests/smoke_tokens.cpp)
    target_link_libraries(test_smoke_tokens PRIVATE mapggml)
    # metric / camera-alignment component parity (benchmarks/eval_parity_unit.py)
    add_executable(test_eval_parity tests/eval_parity.cpp)
    target_link_libraries(test_eval_parity PRIVATE querysplat_core)
    # render_backward parity harness (benchmarks/render_backward_parity.py)
    add_executable(test_render_bwd tests/render_bwd_dump.cpp)
    target_link_libraries(test_render_bwd PRIVATE querysplat_core)
    # TTO component parity (benchmarks/tto_components_parity.py)
    add_executable(test_tto_components tests/test_tto_components.cpp)
    target_link_libraries(test_tto_components PRIVATE querysplat_core)
    # TTO session probe (t4): linear-probe gradient/AdamW check
    add_executable(test_tto_session tests/test_tto_session.cpp)
    target_link_libraries(test_tto_session PRIVATE querysplat_core)
    # t7: finite-difference verification of the decoder backward (graph B)
    add_executable(test_tto_fd tests/test_tto_fd.cpp)
    target_link_libraries(test_tto_fd PRIVATE querysplat_core)
    # C API smoke: load the f16 pair, run_frames S=2, check the result contract
    add_executable(test_querysplat_capi tests/test_querysplat_capi.cpp)
    target_link_libraries(test_querysplat_capi PRIVATE querysplat_capi)
endif()
