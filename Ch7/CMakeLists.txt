include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
    Core    
    Support
    nativecodegen
    OrcJIT
)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh7CombineIncGen)

add_llvm_executable(toyc-ch7
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/LowerToAffineLoops.cpp
    mlir/LowerToLLVM.cpp
    mlir/ShapeInferencePass.cpp
    mlir/ToyCombine.cpp
    
    DEPENDS
    ToyCh7OpsIncGen
    ToyCh7ShapeInferenceInterfaceIncGen
    ToyCh7CombineIncGen
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(toyc-ch7
    PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
)
