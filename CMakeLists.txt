cmake_minimum_required(VERSION 3.18)
project("diffusers-cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/MP>
        $<$<COMPILE_LANGUAGE:C>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(APPLE)
    function(diffusers_set_macos_rpaths target)
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "EXECUTABLE")
            set(runtime_paths "@executable_path" "@executable_path/../lib")
        elseif(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
            set(runtime_paths "@loader_path" "@loader_path/../lib")
            set_target_properties(${target} PROPERTIES
                MACOS_RPATH ON
                INSTALL_NAME_DIR "@rpath"
                BUILD_WITH_INSTALL_NAME_DIR ON
            )
        else()
            return()
        endif()

        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${runtime_paths}"
            INSTALL_RPATH "${runtime_paths}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endfunction()

    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# ---------------------------------------------------------------------------
# GPU backend options (forwarded to ggml)
# ---------------------------------------------------------------------------
option(DIFFUSERS_CUDA          "diffusers: enable CUDA backend"          OFF)
option(DIFFUSERS_HIP           "diffusers: enable HIP/ROCm backend"      OFF)
option(DIFFUSERS_METAL         "diffusers: enable Metal backend"         ${APPLE})
option(DIFFUSERS_VULKAN        "diffusers: enable Vulkan backend"        OFF)
option(DIFFUSERS_OPENCL        "diffusers: enable OpenCL backend"        OFF)
option(DIFFUSERS_SYCL          "diffusers: enable SYCL backend"          OFF)
option(DIFFUSERS_MUSA          "diffusers: enable MUSA backend"          OFF)

if(DIFFUSERS_CUDA)
    message(STATUS "diffusers: enabling CUDA backend")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

if(DIFFUSERS_HIP)
    message(STATUS "diffusers: enabling HIP/ROCm backend")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(DIFFUSERS_METAL)
    message(STATUS "diffusers: enabling Metal backend")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(DIFFUSERS_VULKAN)
    message(STATUS "diffusers: enabling Vulkan backend")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(DIFFUSERS_OPENCL)
    message(STATUS "diffusers: enabling OpenCL backend")
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
endif()

if(DIFFUSERS_SYCL)
    message(STATUS "diffusers: enabling SYCL backend")
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
    if(WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
endif()

if(DIFFUSERS_MUSA)
    message(STATUS "diffusers: enabling MUSA backend")
    set(GGML_MUSA ON CACHE BOOL "" FORCE)
endif()

# ---------------------------------------------------------------------------
# ggml dependency
# ---------------------------------------------------------------------------
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(NOT TARGET ggml)
    if(DIFFUSERS_USE_SYSTEM_GGML)
        find_package(ggml REQUIRED)
        if(NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            message(FATAL_ERROR "ggml submodule not initialized. Run:\n  git submodule update --init")
        endif()
        add_subdirectory(ggml)
    endif()
endif()

# ---------------------------------------------------------------------------
# diffusers library
# ---------------------------------------------------------------------------

file(GLOB DIFFUSERS_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/ggml/*.cpp"
    "src/nn/*.cpp"
    "src/nn/attention/*.cpp"
    "src/nn/modules/**/*.cpp"
    "src/diffusers/models/**/*.cpp"
    "src/diffusers/models/transformers/**/*.cpp"
)

add_library(diffusers STATIC ${DIFFUSERS_SOURCES})

target_include_directories(diffusers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(diffusers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_link_libraries(diffusers PUBLIC ggml)

target_compile_features(diffusers PUBLIC c_std_11 cxx_std_17)

if(APPLE)
    diffusers_set_macos_rpaths(diffusers)
endif()

if(DIFFUSERS_SYCL AND DEFINED SYCL_COMPILE_OPTIONS)
    target_compile_options(diffusers PRIVATE ${SYCL_COMPILE_OPTIONS})
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

option(DIFFUSERS_BUILD_TESTS "diffusers: build test utilities" ON)

if(DIFFUSERS_BUILD_TESTS)
    enable_testing()
    find_program(UV_EXECUTABLE uv REQUIRED)

    set(UV_ENV_STAMP ${CMAKE_CURRENT_BINARY_DIR}/uv-env.stamp)

    add_custom_command(
        OUTPUT ${UV_ENV_STAMP}
        COMMAND ${UV_EXECUTABLE} venv --clear
        COMMAND ${UV_EXECUTABLE} pip install -r requirements.txt
        COMMAND ${CMAKE_COMMAND} -E touch ${UV_ENV_STAMP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test
        COMMENT "Creating uv test environment"
    )

    add_custom_target(uv-env ALL
        DEPENDS ${UV_ENV_STAMP}
    )

    add_executable(test-tensor-cli
        test/utils/tensor-cli/main.cpp
    )

    target_include_directories(test-tensor-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test-tensor-cli PRIVATE diffusers)

    add_executable(test-nn-cli
        test/utils/nn-cli/main.cpp
    )

    target_include_directories(test-nn-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test-nn-cli PRIVATE diffusers)

    add_executable(test-diffusers-cli
        test/utils/diffusers-cli/main.cpp
    )

    target_include_directories(test-diffusers-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test-diffusers-cli PRIVATE diffusers)

    # Generate an individual ctest entry for each test
    file(
        GLOB_RECURSE TEST_FILES
        CONFIGURE_DEPENDS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/src/*.py
    )
    list(FILTER TEST_FILES EXCLUDE REGEX "(/|^)__init__\\.py$")
    list(SORT TEST_FILES)

    foreach(TEST_FILE IN LISTS TEST_FILES)
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)

        # Create a unique test name from the relative path
        string(REPLACE "/" "-" TEST_ID "${TEST_FILE}")
        string(REPLACE "\\" "-" TEST_ID "${TEST_ID}")
        string(REGEX REPLACE "\\.py$" "" TEST_ID "${TEST_ID}")

        add_test(
            NAME "${TEST_ID}"
            COMMAND ${UV_EXECUTABLE} run python -m unittest "${TEST_FILE}" -v
        )

        if(TEST_FILE MATCHES "^src/ggml/Tensor/")
            set(CLI_TARGET test-tensor-cli)
        elseif(TEST_FILE MATCHES "^src/nn/")
            set(CLI_TARGET test-nn-cli)
        elseif(TEST_FILE MATCHES "^src/diffusers/")
            set(CLI_TARGET test-diffusers-cli)
        else()
            message(FATAL_ERROR "No CLI target for ${TEST_FILE}")
        endif()

        set_tests_properties("${TEST_ID}" PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
            ENVIRONMENT "CLI=$<TARGET_FILE:${CLI_TARGET}>;PYTHONDONTWRITEBYTECODE=1"
        )
    endforeach()

    message(STATUS "diffusers: tests enabled")
endif()
