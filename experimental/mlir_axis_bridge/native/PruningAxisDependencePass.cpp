// Standalone local MLIR access analyzer for pruning-axis evidence.
//
// This intentionally remains a selected-subgraph evidence tool. It does not
// execute transformations and does not participate in production analysis.

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct Access {
  std::string tensor;
  std::vector<std::string> indices;
  std::vector<std::string> loopIVs;
  std::string kind;
  std::string evidence;
};

std::string valueName(mlir::Value value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.printAsOperand(stream, mlir::OpPrintingFlags());
  stream.flush();
  return text;
}

std::vector<std::string> parentLoopIVs(mlir::Operation *op) {
  std::vector<std::string> ivs;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(parent))
      ivs.push_back(valueName(loop.getInductionVar()));
    else if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(parent))
      ivs.push_back(valueName(loop.getInductionVar()));
  }
  std::reverse(ivs.begin(), ivs.end());
  return ivs;
}

Access makeAccess(mlir::Operation *op, bool isWrite, unsigned memrefOffset,
                  unsigned indexOffset) {
  Access access;
  access.kind = isWrite ? "write" : "read";
  access.tensor = valueName(op->getOperand(memrefOffset));
  for (unsigned i = indexOffset; i < op->getNumOperands(); ++i)
    access.indices.push_back(valueName(op->getOperand(i)));
  access.loopIVs = parentLoopIVs(op);
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  stream.flush();
  access.evidence = text;
  return access;
}

llvm::json::Array stringArray(const std::vector<std::string> &values) {
  llvm::json::Array array;
  for (const auto &value : values)
    array.push_back(value);
  return array;
}

std::vector<std::string> intersection(const std::vector<std::string> &left,
                                      const std::vector<std::string> &right) {
  std::vector<std::string> values;
  for (const auto &value : left)
    if (std::find(right.begin(), right.end(), value) != right.end())
      values.push_back(value);
  return values;
}

std::vector<std::string> difference(const std::vector<std::string> &left,
                                    const std::vector<std::string> &right) {
  std::vector<std::string> values;
  for (const auto &value : left)
    if (std::find(right.begin(), right.end(), value) == right.end())
      values.push_back(value);
  return values;
}

llvm::json::Object relation(unsigned id, const Access &read,
                            const Access &write,
                            const std::vector<std::string> &sourceIndices,
                            const std::vector<std::string> &targetIndices,
                            llvm::StringRef relationKind,
                            llvm::StringRef dependenceKind,
                            const std::string &proof) {
  return llvm::json::Object{
      {"relation_id", "rel_" + std::to_string(id)},
      {"source_tensor", read.tensor},
      {"source_indices", stringArray(sourceIndices)},
      {"target_tensor", write.tensor},
      {"target_indices", stringArray(targetIndices)},
      {"loop_ivs", stringArray(write.loopIVs)},
      {"relation_kind", relationKind},
      {"dependence_kind", dependenceKind},
      {"affine_evidence",
       llvm::json::Array{read.evidence, write.evidence}},
      {"proof", proof},
      {"confidence", "medium"}};
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                   llvm::cl::desc("<input.mlir>"),
                                   llvm::cl::Required);
  llvm::cl::opt<std::string> output("output",
                                    llvm::cl::desc("JSON output path"),
                                    llvm::cl::value_desc("path"));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Local pruning-axis dependence analyzer\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module) {
    llvm::errs() << "failed to parse MLIR file: " << input << "\n";
    return 2;
  }

  std::vector<Access> reads;
  std::vector<Access> writes;
  std::set<std::string> dialects;
  module->walk([&](mlir::Operation *op) {
    if (!op->getName().getDialectNamespace().empty())
      dialects.insert(op->getName().getDialectNamespace().str());
    if (llvm::isa<mlir::affine::AffineLoadOp>(op) ||
        op->getName().getStringRef() == "memref.load")
      reads.push_back(makeAccess(op, false, 0, 1));
    else if (llvm::isa<mlir::affine::AffineStoreOp>(op) ||
             op->getName().getStringRef() == "memref.store")
      writes.push_back(makeAccess(op, true, 1, 2));
  });

  llvm::json::Array relations;
  std::set<std::string> reductions;
  std::set<std::string> preserved;
  std::set<std::string> blocked;
  unsigned relationID = 0;
  for (const auto &write : writes) {
    std::vector<std::vector<std::string>> reducedByRead;
    for (const auto &read : reads) {
      auto common = intersection(read.indices, write.indices);
      auto reduced = difference(read.indices, write.indices);
      reducedByRead.push_back(reduced);
      if (!common.empty()) {
        for (const auto &iv : common)
          preserved.insert(iv);
        relations.push_back(relation(
            relationID++, read, write, common, common, "preserved",
            "access_equivalence",
            "same IVs appear in source and target accesses"));
      }
      if (!reduced.empty()) {
        for (const auto &iv : reduced)
          reductions.insert(iv);
        relations.push_back(relation(
            relationID++, read, write, reduced, {}, "reduced", "reduction",
            "source IVs disappear from the target access inside enclosing loops"));
      }
    }
    if (reducedByRead.size() >= 2) {
      auto mixed = reducedByRead.front();
      for (unsigned i = 1; i < reducedByRead.size(); ++i)
        mixed = intersection(mixed, reducedByRead[i]);
      for (const auto &iv : mixed) {
        blocked.insert(iv);
        relations.push_back(relation(
            relationID++, reads.front(), write, {iv}, {}, "mixed",
            "reduction",
            "IV participates in multiple reads and disappears from the write; "
            "contraction-style channel mixing is possible"));
      }
    }
  }

  std::vector<std::string> dialectValues(dialects.begin(), dialects.end());
  std::vector<std::string> reductionValues(reductions.begin(), reductions.end());
  std::vector<std::string> preservedValues(preserved.begin(), preserved.end());
  std::vector<std::string> blockedValues(blocked.begin(), blocked.end());
  llvm::json::Object root{
      {"mlir_file", input.getValue()},
      {"analysis_tool", "native_mlir_pass"},
      {"dialects_seen", stringArray(dialectValues)},
      {"relations", std::move(relations)},
      {"reductions", stringArray(reductionValues)},
      {"preserved_axes", stringArray(preservedValues)},
      {"blocked_axes", stringArray(blockedValues)},
      {"warnings", llvm::json::Array{}}};

  std::string rendered = llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
  if (output.empty()) {
    llvm::outs() << rendered << "\n";
    return 0;
  }
  std::error_code error;
  llvm::raw_fd_ostream stream(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open output: " << error.message() << "\n";
    return 3;
  }
  stream << rendered << "\n";
  return 0;
}
