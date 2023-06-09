//===-------- Dialect.cpp - Toy IR Dialect registration in MLIR --------===//
//
// This file implements the dialect for the Toy IR: custom type parsing
// and operation verification.
//
//===-------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

//===------------------------------------------===//
// ToyInlinerInterface
//===------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public mlir::DialectInlinerInterface {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    //===------------------------------------------===//
    // Analysis Hooks
    //===------------------------------------------===//

    /// All call operations within toy can be inlined.
    bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                         bool wouldBeCloned) const final {
        return true;
    }

    /// All operations within toy can be inlined.
    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                         mlir::IRMapping &) const final {
        return true;
    }

    /// All functions within toy can be inlined.
    bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                         mlir::IRMapping &) const final {
        return true;
    }

    //===------------------------------------------===//
    // Toy Dialect
    //===------------------------------------------===//

    /// Handle the given inlined terminator(toy.return) by replacing it with a new
    /// operation as necessary.
    void handleTerminator(mlir::Operation *op,
                          llvm::ArrayRef<mlir::Value> valuesToRepl) const final {
        // Only "toy.return" needs to be handled here.
        auto returnOp = llvm::cast<ReturnOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());                  
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated . nullptr
    /// should be returned.
    mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder,
                                               mlir::Value input,
                                               mlir::Type resultType,
                                               mlir::Location conversionLoc)
                                               const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

//===------------------------------------------===//
// Toy Dialect
//===------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
    >();
    addInterfaces<ToyInlinerInterface>();
}

//===------------------------------------------===//
// Toy Operations
//===------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    llvm::SMLoc operandsLoc = parser.getCurrentLocation();
    mlir::Type type;
    if (parser.parseOperandList(operands, /*requireOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and the results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly.
    mlir::Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](mlir::Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type.
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===------------------------------------------===//
// ConstantOp
//===------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so it is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
    auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return mlir::failure();

    result.addTypes(value.getType());
    return mlir::success();
}

/// The `OpAsmPrinter` class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), /*elideAttrs=*/{"value"});
    printer << getValue();
}

/// Verifier for the constant operation. This correponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult ConstantOp::verify() {
    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!resultType)
        return mlir::success();

    // Check that the rank of the attribute type matches the rank of the constant
    // result type.
    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    if (attrType.getRank() != resultType.getRank()) {
        return emitOpError("return type must match the one of the attched value "
                           "attribute: ")
            << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank();dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return emitError(
                "return type shape mismatches its attribute at dimension ")
                << dim << ": " << attrType.getShape()[dim]
                << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

//===------------------------------------------===//
// AddOp
//===------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AddOp, this is required by the shape
/// inference interfaces.
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===------------------------------------------===//
// SubOp
//===------------------------------------------===//

void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult SubOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void SubOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the SubOp, this is required by the shape
/// inference interfaces.
void SubOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===------------------------------------------===//
// CastOp
//===------------------------------------------===//

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    // The inputs must be Tensors with the same element type.
    mlir::TensorType input = llvm::dyn_cast<mlir::TensorType>(inputs.front());
    mlir::TensorType output = llvm::dyn_cast<mlir::TensorType>(outputs.front());
    if (!input || !output || input.getElementType() != output.getElementType())
        return false;
    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
}

//===------------------------------------------===//
// FuncOp
//===------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
    // Dispatch to the FunctionOpInterface provided utility method that parses the
    // function operation.
    auto buildFuncType =
        [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Returns the region on the function operation that is callable.
mlir::Region *FuncOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() {
    return getFunctionType().getResults();
}

/// Returns the argument attributes for all callable region arguments or
/// null if there are none.
mlir::ArrayAttr FuncOp::getCallableArgAttrs() {
    return getArgAttrs().value_or(nullptr);
}

/// Returns the result attributes for all callable region results or
/// null if there are none.
mlir::ArrayAttr FuncOp::getCallableResAttrs() {
    return getResAttrs().value_or(nullptr);
}

//===------------------------------------------===//
// GenericCallOp
//===------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
    // Generic call always returns an unranked Tensor initally.
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee",
                       mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
mlir::CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the
/// call interface.
void GenericCallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
}

/// Get the argument operands to the called function, this required by the
/// call interface.
mlir::Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

//===------------------------------------------===//
// MulOp
//===------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===------------------------------------------===//
// ReturnOp
//===------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attched to the operation definition.
    auto function =cast<FuncOp>((*this)->getParentOp());

    /// Return Ops can only have a single optional operand.
    if (getNumOperands() > 1)
        return emitOpError() << "expect at most 1 return operand";

    /// The operand number and types must match the function signature.
    const auto &results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
        return emitOpError() << "does not return the same number of values ("
                             << getNumOperands() << ") as the enclosing function ("
                             << results.size() << ")";

    // If the operation does not have an input, we are done.
    if (!hasOperand())
        return mlir::success();

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    // Check that the result type of the function matches the operand type.
    if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
        llvm::isa<mlir::UnrankedTensorType>(resultType))
        return mlir::success();

    return emitError() << "type of the return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}

//===------------------------------------------===//
// TransposeOp
//===------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(value);
}

void TransposeOp::inferShapes() {
    auto arrayTy = llvm::cast<mlir::RankedTensorType>(getOperand().getType());
    llvm::SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
    getResult().setType(mlir::RankedTensorType::get(dims, arrayTy.getElementType()));
}

mlir::LogicalResult TransposeOp::verify() {
    auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(getOperand().getType());
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getType());
    if (!inputType || !resultType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    if (!std::equal(inputShape.begin(), inputShape.end(),
                    resultType.getShape().rbegin())) {
        return emitError()
            << "expected result shape to transpose of the input";
    }
    return mlir::success();
}

//===------------------------------------------===//
// TableGen'd op method definitions
//===------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
