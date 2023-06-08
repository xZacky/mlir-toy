//===---- ToyCombine.td - Pattern Match Optimization for Toy - *- tablegen -* ----===//
//
// Difines language-sepecific pattern match optimizations for Toy using
// Declarative Rewrite Rule (DRR) specified using TableGen records.
//
//===-----------------------------------------------------------------------------===//

#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

/// Note: The DRR definition used for defining patterns is below:
///
/// class Pattern<
///     dag sourcePattern, list<dag> resultPatterns,
///     list<dag> additionalConstraints = [],
///     dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------===//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

//===----------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------===//

// Native Code Calls may be used for more complex transforms using inline
// C++ and C++ helper function.

// Reshape(Constant(x)) = x
def ReshapeConstant :
    NativeCodeCall<"$0.reshape(::llvm::cast<mlir::ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))>;

//===----------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------===//

// DRR allows for  constraint checking when the transformation is conditional
// on operand properties.

// Reshape(x) = x, where input and output shapes are identical
def TypeAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg), (replaceWithValue $arg),
    [(TypeAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE