cmake_minimum_required(VERSION 3.20.0)
project(toyc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS -fPIC -std=gnu++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
# include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
# add_subdirectory(Ch7)
