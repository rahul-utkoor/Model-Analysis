// Optional native MLIR scaffold for selected-subgraph pruning-axis evidence.
//
// This is deliberately out-of-tree and not wired into production analysis.
// TODO: add pass-plugin registration and JSON emission once the local MLIR API
// version is fixed for a durable build contract.

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace {

struct PruningAxisDependencePass
    : public mlir::PassWrapper<PruningAxisDependencePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruningAxisDependencePass)

  llvm::StringRef getArgument() const final {
    return "pruning-axis-dependence";
  }

  llvm::StringRef getDescription() const final {
    return "Collect local affine/scf access evidence for pruning-axis analysis";
  }

  void runOnOperation() final {
    getOperation().walk([&](mlir::func::FuncOp function) {
      llvm::errs() << "[pruning-axis-dependence] function "
                   << function.getSymName() << "\n";
      function.walk([&](mlir::Operation *op) {
        if (llvm::isa<mlir::affine::AffineForOp, mlir::scf::ForOp,
                      mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                      mlir::memref::LoadOp, mlir::memref::StoreOp>(op)) {
          llvm::errs() << "  " << op->getName() << "\n";
        }
      });
    });
  }
};

} // namespace
