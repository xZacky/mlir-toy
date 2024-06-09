add_subdirectory(include)

add_llvm_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp

  DEPENDS
  ToyCh2OpsIncGen
)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(toyc-ch2
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
)
