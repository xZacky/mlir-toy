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

namespace mlir {

namespace toy {

namespace detail {

struct StructTypeStorage;

} // namespace detail

} // namespace toy

} // namespace mlir

/// Include the auto-generated header file containing the declaration of the toy
/// dialect
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declaration of the toy
/// operations
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

namespace mlir {

namespace toy {

//===--------------------------------------------------===//
// Toy Types
//===--------------------------------------------------===//

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Create an instance of a `StructType` with the given element types.
    /// *must* be atleast one element type.
    static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

    /// Returns the element types of this struct type.
    llvm::ArrayRef<mlir::Type> getElementTypes();

    /// Returns the number of element type held by this struct.
    size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // namespace toy

} // namespace mlir

#endif // TOY_DIALECT_H