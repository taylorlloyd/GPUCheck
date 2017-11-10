#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"

#ifndef ADDRSPACE_H
#define ADDRSPACE_H

using namespace std;
using namespace llvm;

namespace gpucheck {

  class AddrSpaceAnalysis : public ModulePass {
  public:
    static char ID;
    AddrSpaceAnalysis() : ModulePass(ID) {}
    bool runOnModule(Module &M);
    void getAnalysisUsage(AnalysisUsage &AU) const;
    bool mayBeGlobal(Value *v);
  };
}
#endif
