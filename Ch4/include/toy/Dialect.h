//===------ Dialect.h - Dialect definition for the Toy IR ------===//
//
// This file implements the IR Dialect for the Toy language.
// 
//===-----------------------------------------------------------===//

#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declaration of the toy
/// operations
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif // TOY_DIALECT_H