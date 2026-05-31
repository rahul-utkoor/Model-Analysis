cmake_minimum_required(VERSION 3.20)
project(PruningAxisDependencePass LANGUAGES CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

add_mlir_library(PruningAxisDependencePass MODULE
  PruningAxisDependencePass.cpp
  LINK_LIBS PRIVATE
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
)

target_compile_features(PruningAxisDependencePass PRIVATE cxx_std_17)
