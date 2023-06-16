//===---------------- Passes.h - Toy Passes Definition ----------------===//
//
// This file exposes the entry points to create compiler passes for toy.
//
//===------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {

class Pass;

namespace toy {

std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `Toy` operations, as well
/// as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace toy

} // namespace mlir

#endif // TOY_PASSES_H
