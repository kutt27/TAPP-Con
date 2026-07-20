function(add_tapp_con_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            tapp_con
            doctest::doctest
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_tapp_con_test(test_constrained_tensor   test_constrained_tensor.cpp)
add_tapp_con_test(test_u1_symmetry          test_u1_symmetry.cpp)
add_tapp_con_test(test_su2_symmetry         test_su2_symmetry.cpp)
add_tapp_con_test(test_block_gemm           test_block_gemm.cpp)
add_tapp_con_test(test_tapp_c_api           test_tapp_c_api.cpp)
add_tapp_con_test(test_constraint_engine    test_constraint_engine.cpp)
add_tapp_con_test(test_threaded_gemm        test_threaded_gemm.cpp)
add_executable(test_tnqvm_adapter
    test_tnqvm_adapter.cpp
    ${CMAKE_SOURCE_DIR}/examples/tnqvm_adapter/tnqvm_tapp_backend.cpp
)
target_link_libraries(test_tnqvm_adapter
    PRIVATE
        tapp_con
        doctest::doctest
)
target_include_directories(test_tnqvm_adapter
    PRIVATE
        ${CMAKE_SOURCE_DIR}/examples/tnqvm_adapter
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_tnqvm_adapter PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME test_tnqvm_adapter COMMAND test_tnqvm_adapter)

# GPU tests — only compiled when CUDA is available
if(CUDAToolkit_FOUND)
    add_executable(test_gpu_gemm test_gpu_gemm.cu)
    target_link_libraries(test_gpu_gemm
        PRIVATE
            tapp_con
            doctest::doctest
            CUDA::cudart
            CUDA::cublas
    )
    add_test(NAME test_gpu_gemm COMMAND test_gpu_gemm)
endif()
