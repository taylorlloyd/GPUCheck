
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"

#include "ThreadDepAnalysis.h"
#include "OffsetPropagation.h"

#ifndef BRANCH_DIVERGE_H
#define BRANCH_DIVERGE_H

namespace gpucheck {

  class BranchDivergeAnalysis : public ModulePass {
    public:
      static char ID;
      BranchDivergeAnalysis() : ModulePass(ID) {}
      void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<ThreadDependence>();
        AU.addRequired<OffsetPropagation>();
        AU.setPreservesAll();
      }
      bool runOnModule(Module &M);
      bool runOnKernel(Function &F);
      float getDivergence(BranchInst *BI);
    private:
      ThreadDependence *TD;
      OffsetPropagation *OP;

  };

}

#endif
