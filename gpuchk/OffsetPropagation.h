#include "OffsetVal.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include <unordered_map>
#include <vector>

#ifndef OFFSET_PROP_H
#define OFFSET_PROP_H

using namespace llvm;

namespace gpucheck {

  class OffsetPropagation : public ModulePass {
    private:
      Module *M;
      std::unordered_map<Value *, OffsetValPtr> offsets;

      OffsetValPtr getOrCreateVal(BinaryOperator *);
      OffsetValPtr getOrCreateVal(CallInst *);
      OffsetValPtr getOrCreateVal(CastInst *);
      OffsetValPtr getOrCreateVal(CmpInst *);
      OffsetValPtr getOrCreateVal(Constant *);
      OffsetValPtr getOrCreateVal(LoadInst *);
      OffsetValPtr getOrCreateVal(PHINode *);
      OffsetValPtr getOrCreateVal(GetElementPtrInst *);
      OffsetValPtr getOrCreateGEPVal(ConstantExpr *);
      OffsetValPtr getGEPExpr(Value *, Type *, Use *, Use *);

      OffsetOperator fromBinaryOpcode(llvm::Instruction::BinaryOps);
      OffsetOperator fromCmpPredicate(llvm::CmpInst::Predicate);

      OffsetValPtr applyDominatingCondition(std::vector<Value *>& values,
                                                std::vector<BasicBlock *>& blocks,
                                                Instruction *mergePt,
                                                DominatorTree& DT);

      std::vector<const Function*> findRequiredContexts(const OffsetValPtr& ptr,
          std::vector<const Function*> found = std::vector<const Function*>());
      std::vector<const CallInst*> getSameModuleFunctionCallers(const Function *f);

      void invalidateRange(Value *start, OffsetValPtr& to);

      std::vector<OffsetValPtr> inContexts(OffsetValPtr& orig, std::vector<const Function*>& ignore);

      bool isUpdateStore(StoreInst* s);

    public:
      static char ID;
      OffsetPropagation() : ModulePass(ID) {}
      bool runOnModule(Module &F);
      void getAnalysisUsage(AnalysisUsage &AU) const;

      OffsetValPtr getOrCreateVal(Value *);
      OffsetValPtr inCallContext(const OffsetValPtr& orig, const CallInst *ci);
      OffsetValPtr inThreadContext(const OffsetValPtr& orig,
          int thread_idx, int thread_idy, int thread_idz,
          int block_idx, int block_idy, int block_idz);
      OffsetValPtr inGridContext(const OffsetValPtr& orig,
          int thread_dimx, int thread_dimy, int thread_dimz,
          int block_dimx, int block_dimy, int block_dimz);

      std::vector<OffsetValPtr> inContexts(OffsetValPtr& orig);
  };
}

#endif
