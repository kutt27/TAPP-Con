add_library(tnqvm_tapp_backend
    tnqvm_tapp_backend.cpp
)

target_include_directories(tnqvm_tapp_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tnqvm_tapp_backend
    PUBLIC
        tapp_con
)

install(TARGETS tnqvm_tapp_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
