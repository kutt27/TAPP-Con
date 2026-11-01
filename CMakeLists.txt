cmake_minimum_required(VERSION 3.20)
project(tapp-con VERSION 0.1.0 LANGUAGES C CXX)

# ── C++ Standard ──
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
option(TAPP_CON_BUILD_TESTS    "Build tests"    ON)
option(TAPP_CON_BUILD_EXAMPLES "Build examples" OFF)
option(TAPP_CON_USE_CUDA       "Enable CUDA"    ON)

# ── Dependencies ──
include(FetchContent)

# BLAS is optional
find_package(BLAS QUIET)

# OpenMP for threaded block GEMM (Phase 2)
find_package(OpenMP QUIET)

# CUDA Toolkit (Phase 3)
if(TAPP_CON_USE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

if(TAPP_CON_BUILD_TESTS)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
    )
    FetchContent_MakeAvailable(doctest)
    enable_testing()
endif()

# ── Library target ──
add_library(tapp_con
    src/tapp_constrained.cpp
    src/constrained_tensor.cpp
    src/symmetry/u1_handler.cpp
    src/symmetry/su2_handler.cpp
    src/backend/block_gemm.cpp
    src/backend/gpu_backend.cpp
    src/constraint/linear_constraint.h
    src/constraint/constraint_parser.cpp
    src/constraint/constraint_solver.cpp
    src/constraint/constraint_engine.cpp
)

target_include_directories(tapp_con
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    PRIVATE
)

if(BLAS_FOUND)
    target_compile_definitions(tapp_con PRIVATE TAPP_CON_USE_CBLAS)
    target_link_libraries(tapp_con PUBLIC ${BLAS_LIBRARIES})
    message(STATUS "BLAS found — using cblas for block GEMM")
else()
    message(STATUS "BLAS not found — using naive triple-loop fallback for block GEMM")
endif()

if(OpenMP_CXX_FOUND)
    target_compile_definitions(tapp_con PRIVATE TAPP_CON_USE_OPENMP)
    target_link_libraries(tapp_con PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — threaded block GEMM enabled")
else()
    message(STATUS "OpenMP not found — single-threaded block GEMM")
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    target_compile_definitions(tapp_con PRIVATE TAPP_CON_USE_CUDA)
    target_link_libraries(tapp_con PUBLIC CUDA::cudart CUDA::cublas)
    target_sources(tapp_con PRIVATE src/backend/gpu_gemm.cu)
    set_target_properties(tapp_con PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "CUDA found — GPU block GEMM enabled")
else()
    message(STATUS "CUDA not found — GPU block GEMM disabled")
endif()

# ── Install rules ──
install(TARGETS tapp_con
    EXPORT tapp_con_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT tapp_con_targets DESTINATION lib/cmake/tapp_con)

# ── Tests ──
if(TAPP_CON_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ── Reference tools ──
# Parity dumper consumed by the Python port's test suite.
if(TAPP_CON_BUILD_TESTS)
    add_executable(dump_reference tools/dump_reference.cpp)
    target_link_libraries(dump_reference PRIVATE tapp_con)
endif()

# ── Examples (Phase 4) ──
if(TAPP_CON_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
