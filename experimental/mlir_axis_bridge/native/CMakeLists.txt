cmake_minimum_required(VERSION 3.20)
project(PruningAxisDependenceTool LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

add_llvm_executable(pruning-axis-dependence
  PruningAxisDependencePass.cpp
)

target_link_libraries(pruning-axis-dependence PRIVATE
  MLIRAffineDialect
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRSCFDialect
  MLIRMemRefDialect
)

target_compile_features(pruning-axis-dependence PRIVATE cxx_std_17)
