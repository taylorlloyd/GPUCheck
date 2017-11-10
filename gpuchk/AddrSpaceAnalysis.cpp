#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/Operator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include "Utilities.h"
#include "AddrSpaceAnalysis.h"

using namespace std;
using namespace llvm;
using namespace gpucheck;

#define DEBUG_TYPE "gpuaddr"

namespace {
  // NVPTX-defined namespaces
  enum AddrSpace {
    Generic = 0,
    Global = 1,
    Internal = 2,
    Shared = 3,
    Constant = 4,
    Local = 5
  };
}

void AddrSpaceAnalysis::getAnalysisUsage(AnalysisUsage& AU) const {
  AU.setPreservesAll();
}

bool AddrSpaceAnalysis::runOnModule(Module &M) {
  DEBUG(
  for(auto F=M.begin(),e=M.end();F!=e; ++F) {
    for(auto B=F->begin(),e=F->end();B!=e; ++B) {
      for(auto I=B->begin(),e=B->end();I!=e; ++I) {
        if(isa<LoadInst>(&*I) || isa<StoreInst>(&*I)) {
          errs() << *I << "\n";
          errs() << "|- mayBeGlobal: " << (mayBeGlobal(&*I) ? "True" : "False") << "\n";
        }
      }
    }
  });
  // Lazy Analysis
  return false;
}

bool AddrSpaceAnalysis::mayBeGlobal(Value *v) {
  // Attempt to dig down to the base
  if(auto L=dyn_cast<LoadInst>(v)) {
    return mayBeGlobal(L->getPointerOperand());
  }
  if(auto S=dyn_cast<StoreInst>(v)) {
    return mayBeGlobal(S->getPointerOperand());
  }
  if(auto OP=dyn_cast<Operator>(v)) {
    if(OP->getOpcode() == Instruction::AddrSpaceCast)
      return mayBeGlobal(OP->getOperand(0));
  }
  if(auto GEP=dyn_cast<GetElementPtrInst>(v)) {
    return mayBeGlobal(GEP->getPointerOperand());
  }


  // Simple stack allocation (local)
  if(auto AI=dyn_cast<AllocaInst>(v)) {
    if(auto pty=dyn_cast<PointerType>(AI->getType())) {
      if(!pty->getElementType()->isPointerTy())
        return false;
    }
  }

  // Address space encoded on the type
  if(v->getType()->isPointerTy()) {
    unsigned addr = v->getType()->getPointerAddressSpace();
    if(addr == AddrSpace::Global || addr == AddrSpace::Constant) {
      return true;
    }
    if(addr == AddrSpace::Shared) {
      return false;
    }
  }

  // If we can't tell, assume it may
  return true;
}

char AddrSpaceAnalysis::ID = 0;
static RegisterPass<AddrSpaceAnalysis> X("gpuaddr", "GPU Address Space Analysis",
                                        false,
                                        true);

#undef DEBUG_TYPE
