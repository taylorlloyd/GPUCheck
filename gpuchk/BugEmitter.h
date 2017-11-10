#include "llvm/IR/Instruction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>

#ifndef BUGEMITTER_H
#define BUGEMITTER_H

using namespace llvm;
using namespace std;

namespace gpucheck {
  enum Severity {
    SEV_UNKNOWN,
    SEV_MIN,
    SEV_MED,
    SEV_MAX
  };
  extern void emitWarning(string warning, Instruction* i, Severity sev);
}
#endif
