#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"

#include "MemCoalesceAnalysis.h"
#include "BugEmitter.h"
#include "Utilities.h"
#include "OffsetOps.h"

#include <vector>
#include <utility>
#include <string>
#include <iostream>

using namespace std;
using namespace llvm;
using namespace gpucheck;

#define DEBUG_TYPE "coalesce"

#define COALESCE_THRES 4.0f
#define ACCESS_SIZE 256

bool MemCoalesceAnalysis::runOnModule(Module &M) {
  TD = &getAnalysis<ThreadDependence>();
  OP = &getAnalysis<OffsetPropagation>();
  ASA = &getAnalysis<AddrSpaceAnalysis>();
  // Run over each GPU function
  for(auto f=M.begin(), e=M.end(); f!=e; ++f) {
    //if(isKernelFunction(*f))
      runOnKernel(*f);
  }
  return false;
}

bool MemCoalesceAnalysis::runOnKernel(Function &F) {

  for(auto b=F.begin(),e=F.end(); b!=e; ++b) {
    for(auto i=b->begin(),e=b->end(); i!=e; ++i) {
      if(auto L=dyn_cast<LoadInst>(i))
        testLoad(L);

      if(auto S=dyn_cast<StoreInst>(i))
        testStore(S);

      if(auto CI=dyn_cast<CallInst>(i))
        testCall(CI);
    }
  }
  return false;
}

void MemCoalesceAnalysis::testCall(CallInst *CI) {
  if(auto MC = dyn_cast<MemCpyInst>(CI))
    if(!testAccess(MC, MC->getDest()))
      testAccess(MC, MC->getSource());

  if(auto MM = dyn_cast<MemMoveInst>(CI))
    if(!testAccess(MM, MM->getDest()))
      testAccess(MM, MM->getSource());
}

void MemCoalesceAnalysis::testLoad(LoadInst *L) {
  testAccess(L, L->getPointerOperand());
}

void MemCoalesceAnalysis::testStore(StoreInst *S) {
  testAccess(S, S->getPointerOperand());
}

bool MemCoalesceAnalysis::testAccess(Instruction *i, Value *ptr) {

  if(!TD->isDependent(ptr))
    return false;
  // Ignore stack allocations
  if(isa<AllocaInst>(ptr))
    return false;
  // Ignore shared/constant memory accesses
  if(!ASA->mayBeGlobal(i))
    return false;
  MemAccess tpe = getAccessType(i, ptr);
  if(tpe == Update && isa<StoreInst>(i)) {
    // Don't report updates twice
    return false;
  }
  DEBUG(errs() << "Found a memory access:\n");
  DEBUG(i->dump());
  DEBUG(errs() << "\n Memory requests required per warp: " <<
      requestsPerWarp(ptr) << "\n");
  // We have a memory access to inspect
  float requests = requestsPerWarp(ptr);
  if(requests > COALESCE_THRES) {
    Severity sev;
    emitWarning(getWarning(&*ptr, tpe, requests, sev), &*i, sev);
    return true;
  }

  return false;
}

MemAccess MemCoalesceAnalysis::getAccessType(Instruction *i, Value *address) {
  bool read = false;
  bool written = false;
  bool memcpy = false;
  for(auto user=address->user_begin(),e=address->user_end(); user != e; ++user) {
    if(auto u=dyn_cast<Instruction>(*user)) {
      if(isa<LoadInst>(u) && u->getParent() == i->getParent())
        read = true;
      if(isa<StoreInst>(u) && u->getParent() == i->getParent())
        written = true;
      if(isa<CallInst>(u) && u->getParent() == i->getParent())
        memcpy = true;
    }
  }
  if(memcpy)
    return MemAccess::Copy;
  if(read && written)
    return MemAccess::Update;
  if(read)
    return MemAccess::Read;
  if(written)
    return MemAccess::Write;
  //TODO: error, should never reach here
  return MemAccess::Unknown;
}

string MemCoalesceAnalysis::getWarning(Value *ptr, MemAccess tpe, float requestsPerWarp, Severity& severity) {
  int reqs = (int) requestsPerWarp;
  string prefix = "";
  switch (tpe) {
    case Write:
      prefix = "In write to "+getValueName(ptr)+", ";
      break;
    case Read:
      prefix = "In read from "+getValueName(ptr)+", ";
      break;
    case Update:
      prefix = "In update to "+getValueName(ptr)+", ";
      break;
    case Copy:
      prefix = "In copy to "+getValueName(ptr)+", ";
      break;
  }

  /*
  APInt *val[1024];
  for(int i = 0; i<1024; i++)
    val[i] = TV->threadDepPortion(ptr, i);

  for(int i = 0; i<1024; i++)
    if(val[i] == nullptr) {
      */
      severity = Severity::SEV_UNKNOWN;
      return prefix + "Possible Uncoalesced Access Detected";
      /*
    }

  APInt stride = *val[1]-*val[0];
  bool strided = true;
  for(int i = 2; i<1024; i++) {
    if(*val[i]-*val[i-1] != stride)
      strided = false;
  }

  if(strided) {
    SmallString<16> strideStr;
    stride.toString(strideStr, 10, true);
    severity = Severity::SEV_MAX;
    return prefix + "Memory access stride " + string(strideStr.c_str()) + " exceeds max stride 4, requires " + to_string(reqs) + " requests";
  }

  // Let's set severity by how uncoalesced the access is
  if(reqs > 16) {
    severity = Severity::SEV_MAX;
  } else if(reqs > 8) {
    severity = Severity::SEV_MED;
  } else {
    severity = Severity::SEV_MIN;
  }
  return prefix + "Uncoalesced Memory Access requires " + to_string(reqs) + " requests/warp";
  */
}

float MemCoalesceAnalysis::requestsPerWarp(Value *ptr) {

  OffsetValPtr ptr_offset = OP->getOrCreateVal(ptr);
  assert(ptr_offset != nullptr);
  DEBUG(errs() << "Analyzing possibly uncoalesced access:\n    " << *ptr << "\n");
  vector<OffsetValPtr> all_paths = OP->inContexts(ptr_offset);
  DEBUG(errs() << "Context-sensitive analysis generated " << all_paths.size() << " contexts\n");

  float maxRequests = 0.0f;
  for(auto path=all_paths.begin(),e=all_paths.end(); path != e; ++path) {
    // Apply some (arbitrary) grid boundaries
    OffsetValPtr gridCtx = OP->inGridContext(*path, 256, 32, 32, 1, 1, 1);
    DEBUG(cerr << "In grid context: " << *gridCtx <<"\n");
    // Perform as much simplification as we can early
    OffsetValPtr simp = simplifyOffsetVal(sumOfProducts(gridCtx));

    // Optimization: Calculate the difference between threads 0 and 1
    OffsetValPtr threadDiff = cancelDiffs(make_shared<BinOpOffsetVal>(
        OP->inThreadContext(simp,1,0,0,0,0,0),
        Sub,
        OP->inThreadContext(simp,0,0,0,0,0,0)), *TD);

    if(!threadDiff->isConst()) {
      DEBUG(errs() << "Cannot generate constant for access. Expression follows.\n");
      DEBUG(cerr << *threadDiff <<"\n");
      auto rnge = threadDiff->constRange();
      DEBUG(errs() << "Range: " << rnge.first << " to " << rnge.second << "\n");
      return 32.0; // Branch cannot be analyzed in at least 1 context
    }


    int requestCount = 0;
    for(int warp=0; warp<8; warp++) {
      OffsetValPtr warpBase = OP->inThreadContext(simp, warp*32, 0, 0, 0, 0, 0);
      vector<std::pair<long long, long long>> requests;
      for(int tid=0; tid<32; tid++) {
        OffsetValPtr threadBase = OP->inThreadContext(simp, warp*32+tid, 0, 0, 0, 0, 0);
        OffsetValPtr threadDiff = cancelDiffs(make_shared<BinOpOffsetVal>(warpBase, Sub, threadBase), *TD);

        if(!threadDiff->isConst()) {
          requestCount++;
          continue;
        }
        long long offset = threadDiff->constVal().getSExtValue();

        bool fits = false;
        for(auto r=requests.begin(),e=requests.end();r!=e;++r) {
          if(offset >= r->first && offset <= r->second) {
            fits = true;
            break;
          } else if(offset < r->first && offset >= r->second - ACCESS_SIZE) {
            r->first = offset;
            fits = true;
            break;
          } else if(offset + 4 > r->second && offset + 4 <= r->first + ACCESS_SIZE) {
            r->second = offset + 4;
            fits = true;
            break;
          }
        }
        if(!fits)
          requests.push_back(make_pair(offset, offset + 4));
      }
      requestCount += requests.size();
    }

    if(requestCount/8.0f > maxRequests) {
      maxRequests = requestCount/8.0f;
      if(maxRequests > COALESCE_THRES) {
        return maxRequests; // Might as well short-circuit here
      }
    }
  }
  return maxRequests / 32.0f;
}

char MemCoalesceAnalysis::ID = 0;
static RegisterPass<MemCoalesceAnalysis> X("coalesce", "Locate uncoalesced memory accesses in GPU code",
                                        false,
                                        true);
