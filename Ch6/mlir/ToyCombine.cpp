//===---------------- ToyCombine.cpp - Toy High Level Optimizer ----------------===//
//
// This file implements as set of simple combiners for optimizaing operations in
// The Toy dialect.
//
//===---------------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>

using namespace mlir::toy;

namespace {

/// Inlcude the patterns defined in the Declarative Rewirite framework.
#include "ToyCombine.inc"

} // namespace

/// This is an example of a c++ rewirte pattern for the TransposeOp. It
/// Optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    /// We register this pattern to match every toy.transpose in the IR.
    /// The "benefit" is used by the framework to order the patterns and process
    /// them in order of profitability.
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
    
    /// This methods attempts to match a pattern and rewirite it. The rewriter
    /// argument is the orchestrator of the sequence of rewrites. The pattern is
    /// expected to interact with it to perform any changes to the IR from here.
    mlir::LogicalResult matchAndRewrite(TransposeOp op,
                                  mlir::PatternRewriter & rewriter) const override {
        // Look through the input of the current transpose.
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        // Input defined by another transpose? If not, no match.
        if (!transposeInputOp)
            return mlir::failure();

        // Otherwise, we have a reduntant transpose. Use the rewriter.
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return mlir::success();
    }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canoncalization framework.
void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              mlir::MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonizalization framework.
void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              mlir::MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                FoldConstantReshapeOptPattern>(context);
}