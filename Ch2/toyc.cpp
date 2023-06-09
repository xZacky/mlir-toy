//===-------------- toyc.cpp - The Toy Compiler --------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===---------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include <memory>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
                                          
namespace {

enum InputType { Toy, MLIR };

} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decide the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as MLIR file")));

namespace {

enum Action { None, DumpAST, DumpMLIR };

} // namespace
static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int dumpMLIR() {
    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context;
    context.getOrLoadDialect<mlir::toy::ToyDialect>();

    // Handle '.toy' input to the compiler.
    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).endswith(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
        if (!module)
            return 1;

        module->dump();
        return 0;
    }

    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = 
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    module->dump();
    return 0;
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;

    dump(*moduleAST);
    return 0;
}

int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

    switch(emitAction) {
        case Action::DumpAST:
            return dumpAST();
        case Action::DumpMLIR:
            return dumpMLIR();
        default:
            llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return 0;
}
