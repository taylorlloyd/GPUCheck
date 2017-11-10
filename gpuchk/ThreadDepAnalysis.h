#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"

#include <queue>
#include <unordered_map>

#ifndef THREAD_DEP_H
#define THREAD_DEP_H

using namespace std;
using namespace llvm;

namespace gpucheck {

  class ThreadDependence : public ModulePass {
  public:
    static char ID;
    ThreadDependence() : ModulePass(ID) {}
    bool runOnModule(Module &M);
    bool runOnFunction(Function &F);
    bool isDependent(Value *v);

    void getAnalysisUsage(AnalysisUsage &AU) const;

  private:
    bool functionTainted(Function &F, unordered_map<Value *, bool>& taintMap);
    void update(Value *v, bool newVal, unordered_map<Value *, bool>& taintMap, queue<Value *>& worklist);
    bool isDependent(Value *v, unordered_map<Value *, bool>& taintMap, DominatorTree *DT);

    unordered_map<Value *,bool> taint;
    unordered_map<CallInst *,unordered_map<Value *, bool>> callTaint;
  };
} // End gpucheck

#endif
