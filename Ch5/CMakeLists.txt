include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5CombineIncGen)

add_llvm_executable(toyc-ch5
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/LowerToAffineLoops.cpp
    mlir/ShapeInferencePass.cpp
    mlir/ToyCombine.cpp
    
    DEPENDS
    ToyCh5OpsIncGen
    ToyCh5ShapeInferenceInterfaceIncGen
    ToyCh5CombineIncGen
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(toyc-ch5
    PRIVATE
        ${dialect_libs}
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransforms
)
