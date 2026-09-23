include(uipc_utils)

set(app_name larva)

add_executable(${app_name})
target_sources(${app_name} PRIVATE "main.cpp")
target_link_libraries(${app_name} PRIVATE uipc::uipc app)
target_compile_definitions(${app_name} PRIVATE CONFIG_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/")


uipc_target_set_relative_source_file(${app_name})
uipc_target_set_output_directory(${app_name})
uipc_target_add_backend_dependency(${app_name})
set_property(TARGET ${app_name} PROPERTY FOLDER "apps/examples")