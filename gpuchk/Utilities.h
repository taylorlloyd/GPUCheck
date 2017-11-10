
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Dominators.h"
#include <vector>
#include <string>

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

using namespace llvm;
using namespace std;

namespace gpucheck {
  extern bool isKernelFunction(const Function& F);
  extern Value *getDominatingCondition(Instruction *l, Instruction *r, DominatorTree *DT);
  extern Value *getDominatingCondition(BasicBlock *l, BasicBlock *r, DominatorTree *DT);
  extern string getValueName(Value *v);
}

#endif
