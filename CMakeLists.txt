cmake_minimum_required(VERSION 3.18)
project(
    tgn
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

option(TGN_BUILD_TESTS "Build test binaries" OFF)
option(TGN_BUILD_EXAMPLES "Build example binaries" OFF)
option(TGN_BUILD_PYTHON "Build Python Bindings" OFF)
option(TGN_BUILD_MODELS "Build the tguf_models library (TGN)" ON)
option(TGN_NATIVE_OPTIMIZATIONS "Use -march=native -ffast-math in Release builds" ON)

if((TGN_BUILD_TESTS OR TGN_BUILD_EXAMPLES) AND NOT TGN_BUILD_MODELS)
    message(FATAL_ERROR "TGN_BUILD_TESTS and TGN_BUILD_EXAMPLES require TGN_BUILD_MODELS=ON")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow                 # Catches local variables hiding class members
    -Wnon-virtual-dtor       # Ensures proper cleanup of polymorphic classes
    -Wold-style-cast         # Encourages static_cast over C-style casts
    -Woverloaded-virtual     # Warns if you accidentally hide virtual functions
    -Wnull-dereference       # Warns if a null dereference is detected
)

include(FetchContent)

if(TGN_BUILD_PYTHON)
    # Build against the torch in the Python env, so the extension matches it at import.
    # No find_package(Torch): for CUDA torch it requires the CUDA toolkit, which pip
    # wheels do not ship. Headers, core libs and the ABI flag are enough.
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import os, torch; r = os.path.dirname(torch.__file__); print(os.path.join(r, 'include')); print(os.path.join(r, 'include', 'torch', 'csrc', 'api', 'include')); print(os.path.join(r, 'lib')); print(int(torch._C._GLIBCXX_USE_CXX11_ABI)); print(torch.__version__)"
        OUTPUT_VARIABLE TGN_TORCH_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TGN_TORCH_RESULT
    )
    if(NOT TGN_TORCH_RESULT EQUAL 0)
        message(FATAL_ERROR "TGUF: Python build requires torch installed in ${Python_EXECUTABLE}")
    endif()
    string(REPLACE "\n" ";" TGN_TORCH_INFO "${TGN_TORCH_INFO}")
    list(GET TGN_TORCH_INFO 0 TGN_TORCH_INCLUDE_DIR)
    list(GET TGN_TORCH_INFO 1 TGN_TORCH_API_INCLUDE_DIR)
    list(GET TGN_TORCH_INFO 2 TGN_TORCH_LIB_DIR)
    list(GET TGN_TORCH_INFO 3 TGN_TORCH_CXX11_ABI)
    list(GET TGN_TORCH_INFO 4 TGN_TORCH_VERSION)
    message(STATUS "TGUF: Using installed torch ${TGN_TORCH_VERSION} (${TGN_TORCH_LIB_DIR})")

    set(TGN_TORCH_LINK_LIBRARIES)
    foreach(lib c10 torch torch_cpu)
        set(lib_path "${TGN_TORCH_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        if(NOT EXISTS "${lib_path}")
            message(FATAL_ERROR "TGUF: ${lib_path} not found in the installed torch")
        endif()
        list(APPEND TGN_TORCH_LINK_LIBRARIES "${lib_path}")
    endforeach()

    add_library(tgn_torch INTERFACE)
    target_include_directories(tgn_torch SYSTEM INTERFACE
        "${TGN_TORCH_INCLUDE_DIR}" "${TGN_TORCH_API_INCLUDE_DIR}")
    target_link_libraries(tgn_torch INTERFACE ${TGN_TORCH_LINK_LIBRARIES})

    set(TORCH_LIBRARIES tgn_torch)
    set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TGN_TORCH_CXX11_ABI}")
else()
    set(CUDA_VERSION "cpu" CACHE STRING "CUDA version (cpu, 12.6, 12.8, 13.0)")

    if(APPLE) # Target Apple Silicon (M1/M2/M3)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip")
        message(STATUS "TGUF: Target System is APPLE (ARM64). Using CPU LibTorch.")
    else() # Target Linux x86_64
        if(CUDA_VERSION STREQUAL "cpu")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip")
            message(STATUS "TGUF: Target System is LINUX (CPU).")
        else()
            # Clean "12.6" -> "126" for the URL mapping
            string(REPLACE "." "" CUDA_TAG ${CUDA_VERSION})
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu${CUDA_TAG}/libtorch-shared-with-deps-2.10.0%2Bcu${CUDA_TAG}.zip")
            message(STATUS "TGUF: Target System is LINUX (CUDA ${CUDA_VERSION}).")
            enable_language(CUDA)
            add_compile_definitions(TGN_WITH_CUDA)
        endif()
    endif()

    message(STATUS "Downloading Libtorch (${LIBTORCH_URL})")
    FetchContent_Declare(
        libtorch
        URL ${LIBTORCH_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    FetchContent_MakeAvailable(libtorch)
    message(STATUS "Downloading Libtorch (${LIBTORCH_URL}) - done")

    set(CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
    find_package(Torch REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "Downloading fmt")
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.4
)
FetchContent_MakeAvailable(fmt)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build: Enabling Address and UndefinedBehavior Sanitizers")
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build: Enabling some Compiler Optimizations")
    add_compile_options(-O3 -pthread)
    if(TGN_NATIVE_OPTIMIZATIONS)
        add_compile_options(-march=native -ffast-math)
    endif()
endif()

# TGUF Target
# Static for Python builds, so the extension module is a single self-contained .so
if(TGN_BUILD_PYTHON)
    set(TGUF_LIB_TYPE STATIC)
else()
    set(TGUF_LIB_TYPE SHARED)
endif()

add_library(tguf
    ${TGUF_LIB_TYPE}
        src/tguf/store.cpp
        src/tguf/tguf.cpp
)
target_link_libraries(tguf
    PUBLIC
        "${TORCH_LIBRARIES}"
        fmt::fmt
)
target_include_directories(tguf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(tguf PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

# TGUF-Models Target
if(TGN_BUILD_MODELS)
    add_library(tguf_models
        SHARED
            src/tguf_models/common/sampler.cpp
            src/tguf_models/common/scatter_ops.cpp
            src/tguf_models/tgn.cpp
    )

    target_link_libraries(tguf_models
        PUBLIC
            tguf
    )

    target_include_directories(tguf_models
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    set_target_properties(tguf_models PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")
endif()

if (TGN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (TGN_BUILD_PYTHON)
    add_subdirectory(python)
endif()


if (TGN_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # TGUF unit tests
    add_executable(test_tguf
        test/unit/tguf/test_store.cpp
        test/unit/tguf/test_tguf.cpp
    )
    target_link_libraries(test_tguf PRIVATE
        tguf
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_tguf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # TGUF-Model unit tests
    add_executable(test_tguf_models
        test/unit/tguf_models/test_recency_sampler.cpp
        test/unit/tguf_models/test_scatter_ops.cpp
        test/unit/tguf_models/test_tgn.cpp
    )
    target_link_libraries(test_tguf_models PRIVATE
        tguf_models
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_tguf_models PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GoogleTest)
    gtest_discover_tests(test_tguf PROPERTIES LABELS "tguf")
    gtest_discover_tests(test_tguf_models PROPERTIES LABELS "models")
endif()
