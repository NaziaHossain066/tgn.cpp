cmake_minimum_required(VERSION 3.16)
project(
    tguf_python
    VERSION 0.1.0
    DESCRIPTION "Python bindings for TGUF storage layer"
    LANGUAGES CXX
)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    _tguf_py
    NB_STATIC
    bind.cpp
)

target_link_libraries(_tguf_py PRIVATE tguf)

# Stubs are committed (tguf/_tguf_py.pyi), regenerate them with `make stubs`

# tguf is linked statically, so LibTorch is the only runtime dependency
# $ORIGIN/../torch/lib: Look in the torch package next to tguf (for installed versions)
# ${TGN_TORCH_LIB_DIR}: Look where the build-time torch lives (for build trees)
if(APPLE)
    set(TGN_TORCH_RPATH "@loader_path/../torch/lib")
else()
    set(TGN_TORCH_RPATH "$ORIGIN/../torch/lib")
endif()

set_target_properties(_tguf_py PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${TGN_TORCH_LIB_DIR}"
    INSTALL_RPATH "${TGN_TORCH_RPATH}"
)

# Record the torch version used at build time (checked on import in tguf/__init__.py)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
     "TORCH_VERSION = \"${TGN_TORCH_VERSION}\"\n")

install(TARGETS _tguf_py
        LIBRARY DESTINATION tguf
        COMPONENT python_bindings)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
        DESTINATION tguf
        COMPONENT python_bindings)
