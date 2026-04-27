cmake_minimum_required(VERSION 4.2.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(vulkan_render) 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_program(SLANGC_COMPILER slangc REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_SOURCE ${SHADER_DIR}/shader.slang)
set(SHADER_OUTPUT ${SHADER_DIR}/slang.spv) 

add_custom_command(
    OUTPUT ${SHADER_OUTPUT}
    COMMAND ${SLANGC_COMPILER} ${SHADER_SOURCE}
        -target spirv
        -profile spirv_1_4
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        -entry vertMain
        -entry fragMain
        -o ${SHADER_OUTPUT}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
)

add_custom_target(CompileShaders ALL
    DEPENDS ${SHADER_OUTPUT}
)

add_executable(
    main
    ${SRC_LIST}
)
add_dependencies(main CompileShaders)

set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

target_include_directories(
    main
    PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(
    main
    PRIVATE
    Vulkan::Vulkan
    glfw
)



