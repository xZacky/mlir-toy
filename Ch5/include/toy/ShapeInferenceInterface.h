//===---- ShapeInferenceInterface.h - Interface definitions for ShapeInference ----===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===------------------------------------------------------------------------------===//

#ifndef TOY_SHAPEINFERENCEINTERFACE_H
#define TOY_SHAPEINFERENCEINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {

namespace toy {

#include "toy/ShapeInferenceOpInterfaces.h.inc"

} // namespace toy

} // namespace mlir

#endif // TOY_SHAPEINFERENCEINTERFACE_H
