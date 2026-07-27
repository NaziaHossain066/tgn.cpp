cmake_minimum_required(VERSION 3.16)
project(
    tgn
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

option(TGN_BUILD_TESTS "Build test binaries" OFF)
option(TGN_BUILD_EXAMPLES "Build example binaries" OFF)
option(TGN_BUILD_PYTHON "Build Python Bindings" OFF)

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
    # Reuse the installed torch's libtorch instead of fetching a second copy
    if(CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        RESULT_VARIABLE TORCH_IMPORT_RESULT
    )
    if(NOT TORCH_IMPORT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "TGN_BUILD_PYTHON is ON but `import torch` failed in "
            "${Python_EXECUTABLE}. Install torch in this environment before "
            "configuring (e.g. `uv run --with torch ...`).")
    endif()
    message(STATUS "TGUF: Using libtorch bundled with the installed torch package (${TORCH_CMAKE_PREFIX})")
    set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
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
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()

find_package(OpenMP REQUIRED)

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
    # Override for portable builds: "native" SIGILLs on hosts lacking the build host's ISA extensions
    set(TGN_ARCH "native" CACHE STRING "Target CPU arch for -march (e.g. native, x86-64-v3, znver2)")
    message(STATUS "Release build: Enabling some Compiler Optimizations (-march=${TGN_ARCH})")
    add_compile_options(-O3 -march=${TGN_ARCH} -ffast-math -pthread)
endif()

# TGUF Target
add_library(tguf
    SHARED
        src/tguf/store.cpp
        src/tguf/tguf.cpp
)
target_link_libraries(tguf
    PUBLIC
        "${TORCH_LIBRARIES}"
        OpenMP::OpenMP_CXX
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
