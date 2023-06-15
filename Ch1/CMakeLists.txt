set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(toyc-ch1
    toyc.cpp
    parser/AST.cpp
)
include_directories(include/)
target_link_libraries(toyc-ch1
    PRIVATE
        MLIRSupport
)
