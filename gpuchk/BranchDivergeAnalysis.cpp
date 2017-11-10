#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"


#include "BranchDivergeAnalysis.h"
#include "BugEmitter.h"
#include "OffsetOps.h"
#include "Utilities.h"

using namespace std;
using namespace llvm;
using namespace gpucheck;

#define DEBUG_TYPE "bdiverge"

#define DIVERGE_THRESH 0.1f


bool BranchDivergeAnalysis::runOnModule(Module &M) {
  TD = &getAnalysis<ThreadDependence>();
  OP = &getAnalysis<OffsetPropagation>();
  // Run over each kernel function
  for(auto f=M.begin(), e=M.end(); f!=e; ++f) {
    if(!f->isDeclaration()) {
      runOnKernel(*f);
    }
  }
  return false;
}

bool BranchDivergeAnalysis::runOnKernel(Function &F) {
  for(auto b=F.begin(),e=F.end(); b!=e; ++b) {

    for(auto i=b->begin(),e=b->end(); i!=e; ++i) {
      if(auto B=dyn_cast<BranchInst>(i)) {
        if(B->isConditional() && TD->isDependent(B)) {
          // We've found a potentially divergent branch!
          // TODO: Determine if branch is high-cost
          float divergence = getDivergence(B);
          if(divergence > DIVERGE_THRESH) {
            emitWarning("Divergent Branch Detected", B, SEV_MED);
            DEBUG(
              errs() << "Found Divergent Branch!! diverge=(" << divergence << ")\n";
              //B->dump();
              errs() << "\n\n";
            );
          } else {
            DEBUG(
              errs() << "Nondivergent branch, diverge=(" << divergence << ")\n";
              //B->dump();
              errs() << "\n\n";
            );
          }
        }
      }
    }
  }
  return false;
}

float BranchDivergeAnalysis::getDivergence(BranchInst *BI) {
  assert(BI->isConditional());

  // Get the symbolic offset for the branch pointer
  OffsetValPtr cond_offset = OP->getOrCreateVal(BI->getCondition());

  DEBUG(errs() << "Analyzing possibly divergent branch condition:\n    " << *BI->getCondition() << "\n");

  vector<OffsetValPtr> all_paths = OP->inContexts(cond_offset);
  DEBUG(errs() << "Context-sensitive analysis generated " << all_paths.size() << " contexts\n");

  float maxDivergence = 0.0f;
  for(auto path=all_paths.begin(),e=all_paths.end(); path != e; ++path) {
    // Apply some (arbitrary) grid boundaries
    OffsetValPtr gridCtx = OP->inGridContext(*path, 256, 32, 32, 1, 1, 1);
    // Perform as much simplification as we can early
    OffsetValPtr simp = simplifyOffsetVal(sumOfProducts(gridCtx));

    // Calculate the difference between threads 0 and 1
    OffsetValPtr threadDiff = cancelDiffs(make_shared<BinOpOffsetVal>(
        OP->inThreadContext(simp,1,0,0,0,0,0),
        Sub,
        OP->inThreadContext(simp,0,0,0,0,0,0)), *TD);

    if(!threadDiff->isConst()) {
      DEBUG(errs() << "Cannot generate constant for branch. Expression follows.\n");
      DEBUG(cerr << *threadDiff <<"\n");
      auto rnge = threadDiff->constRange();
      DEBUG(errs() << "Range: " << rnge.first << " to " << rnge.second << "\n");
      return 1.0; // Branch cannot be analyzed in at least 1 context
    }

    int divergent = 0;
    for(int warp=0; warp<8; warp++) {
      OffsetValPtr warpBase = OP->inThreadContext(simp, warp*32, 0, 0, 0, 0, 0);
      for(int i=1; i<32; i++) {
        OffsetValPtr threadBase = OP->inThreadContext(simp, warp*32+i, 0, 0, 0, 0, 0);
        OffsetValPtr threadDiff = cancelDiffs(make_shared<BinOpOffsetVal>(warpBase, Sub, threadBase), *TD);
        if(!threadDiff->isConst() || threadDiff->constVal() != 0) {
          divergent++;
          break; // We found divergence, we're done with the warp
        }
      }
    }
    if(divergent/8.0f > maxDivergence)
      maxDivergence = divergent/8.0f;
  }
  return maxDivergence;
}

char BranchDivergeAnalysis::ID = 0;
static RegisterPass<BranchDivergeAnalysis> X("bdiverge", "Locate divergent branches in GPU code",
                                        false,
                                        true);

#undef DEBUG_TYPE
