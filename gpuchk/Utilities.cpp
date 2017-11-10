#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "Utilities.h"

#define DEBUG_TYPE "gpuutil"

bool gpucheck::isKernelFunction(const Function &F) {
    NamedMDNode *NMD = F.getParent()->getNamedMetadata("nvvm.annotations");
    if(NMD) {
      for(auto n=NMD->op_begin(),e=NMD->op_end(); n!=e; ++n) {
        if((*n)->getNumOperands() > 2)
          if(auto str = dyn_cast<MDString>((*n)->getOperand(1))) {
            if(str->getString() == "kernel") {
              const Function *k = mdconst::dyn_extract_or_null<Function>((*n)->getOperand(0));
              if(&F == k)
                return true;
            }
          }
      }
    }
    return F.getCallingConv() == CallingConv::PTX_Kernel;
}

Value *gpucheck::getDominatingCondition(Instruction *left, Instruction *right, DominatorTree *DT) {
 return getDominatingCondition(left->getParent(), right->getParent(), DT);
}

Value *gpucheck::getDominatingCondition(BasicBlock *left, BasicBlock *right, DominatorTree *DT) {
  BasicBlock *dom = DT->findNearestCommonDominator(left, right);

  if(dom->size() == 0)
    return nullptr;
  auto last = --(dom->end());
  if(auto B=dyn_cast<BranchInst>(&*last)) {
    if(B->isConditional())
      return B->getCondition();
  }
  return nullptr;
}

string gpucheck::getValueName(Value *v) {
  // Constants can always generate themselves
  if(auto C=dyn_cast<ConstantInt>(v)) {
    SmallString<16> cint;
    C->getValue().toString(cint, 10, true);
    return string(cint.c_str());
  }

  // Need a function from here on out
  Function* F = nullptr;
  if(auto arg=dyn_cast<Argument>(v))
    F=arg->getParent();
  if(auto i=dyn_cast<Instruction>(v))
    F=i->getParent()->getParent();

  if(F == nullptr)
    return "tmp";

  for(auto i=inst_begin(F),e=inst_end(F); i!=e; ++i) {
    if(auto decl = dyn_cast<DbgDeclareInst>(&*i)) {
      if(decl->getAddress() == v) return decl->getVariable()->getName();
    } else if(auto val = dyn_cast<DbgValueInst>(&*i)) {
      if(val->getValue() == v) return val->getVariable()->getName();
    }
  }

  if(auto GEP=dyn_cast<GetElementPtrInst>(v)) {
    string base = getValueName(GEP->getPointerOperand());
    if(GEP->getNumIndices() > 0) {
      string offset = getValueName(*GEP->idx_begin());
      return base + "[" + offset + "]";
    } else {
      return "*" + base;
    }
  }
  if(auto L=dyn_cast<LoadInst>(v)) {
    return getValueName(L->getPointerOperand());
  }
  if(auto BO=dyn_cast<BinaryOperator>(v)) {
    string left = getValueName(BO->getOperand(0));
    string right = getValueName(BO->getOperand(1));
    switch(BO->getOpcode()) {
    case BinaryOperator::Add:
      return left + "+" + right;
    case BinaryOperator::Sub:
      return left + "-" + right;
    case BinaryOperator::Mul:
      return left + "*" + right;
    case BinaryOperator::SDiv:
    case BinaryOperator::UDiv:
      return left + "/" + right;
    case BinaryOperator::AShr:
    case BinaryOperator::LShr:
      return left + ">>" + right;
    case BinaryOperator::Shl:
      return left + "<<" + right;
    case BinaryOperator::And:
      return left + "&&" + right;
    case BinaryOperator::Or:
      return left + "||" + right;
    case BinaryOperator::Xor:
      return left + "^" + right;
    default:
      break;
    }
  }
  if(auto C=dyn_cast<CastInst>(v)) {
    return getValueName(C->getOperand(0));
  }
  if(auto CI=dyn_cast<CallInst>(v)) {
    if(auto F=CI->getCalledFunction()) {
      switch(F->getIntrinsicID()) {
        case Intrinsic::nvvm_read_ptx_sreg_tid_x:
          return "threadIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_tid_y:
          return "threadIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_tid_z:
          return "threadIdx.z";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
          return "threadDim.x";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
          return "threadDim.y";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
          return "threadDim.z";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
          return "blockIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
          return "blockIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
          return "blockIdx.z";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
          return "blockDim.x";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
          return "blockDim.y";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
          return "blockDim.z";
        case Intrinsic::nvvm_read_ptx_sreg_laneid:
          return "laneID";
        default:
          break;
      }
    }
  }

  DEBUG(errs() << "Unrecognized instruction: "; v->dump(););
  return "tmp";
}
#undef DEBUG_TYPE
