#include "OffsetPropagation.h"
#include "OffsetOps.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/User.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "acf"

STATISTIC(ACFTranslations, "Number of ACF Expressions/Subexpressions Generated");
STATISTIC(ACFBinOpTranslations, "Number of BinOp ACF Expressions Generated");
STATISTIC(ACFCallTranslations, "Number of Call ACF Expressions Generated");
STATISTIC(ACFCastTranslations, "Number of Cast ACF Expressions Generated");
STATISTIC(ACFCmpTranslations, "Number of Cmp ACF Expressions Generated");
STATISTIC(ACFLoadTranslations, "Number of Load ACF Expressions Generated");
STATISTIC(ACFPhiTranslations, "Number of Phi ACF Expressions Generated");
STATISTIC(ACFGEPTranslations, "Number of GEP ACF Expressions Generated");
STATISTIC(ACFArgTranslations, "Number of Arg ACF Expressions Generated");
STATISTIC(ACFUnkInstTranslations, "Number of Unknown Instruction ACF Expressions Generated");
STATISTIC(MaxIACFSize, "Maximum IACF Set Size");

namespace gpucheck {
  using namespace std;

  void OffsetPropagation::getAnalysisUsage(AnalysisUsage& AU) const {
    AU.setPreservesAll();
    AU.addRequired<MemoryDependenceWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool OffsetPropagation::runOnModule(Module &M) {
    // Save my owned module
    this->M = &M;
    // Empty any calculated results
    this->offsets.clear();

    // OffsetVals are evaluated lazily as required
    return false;
  }

  /**
   * Generic method for any value, used to dispatch to the others
   */
  OffsetValPtr OffsetPropagation::getOrCreateVal(Value *v) {
    if(offsets.count(v))
      return offsets[v];
    ++ACFTranslations;
    if(auto b = dyn_cast<BinaryOperator>(v)) return getOrCreateVal(b);
    if(auto c = dyn_cast<CallInst>(v)) return getOrCreateVal(c);
    if(auto c = dyn_cast<CastInst>(v)) return getOrCreateVal(c);
    if(auto c = dyn_cast<CmpInst>(v)) return getOrCreateVal(c);
    if(auto l = dyn_cast<LoadInst>(v)) return getOrCreateVal(l);
    if(auto p = dyn_cast<PHINode>(v)) return getOrCreateVal(p);
    if(auto g = dyn_cast<GetElementPtrInst>(v)) return getOrCreateVal(g);
    if(auto ce = dyn_cast<ConstantExpr>(v)) {
      if(ce->getOpcode() == Instruction::GetElementPtr) {
        return getOrCreateGEPVal(ce);
      }
    }
    if(auto c = dyn_cast<ConstantInt>(v)) return getOrCreateVal(c);

    // Fallthrough, unknown instruction
    if(auto i=dyn_cast<Instruction>(v)) {
      ++ACFUnkInstTranslations;
      offsets[v] = make_shared<InstOffsetVal>(i);
      return offsets[v];
    } else if(auto a=dyn_cast<Argument>(v)) {
      ++ACFArgTranslations;
      offsets[v] = make_shared<ArgOffsetVal>(a);
      return offsets[v];
    } else {
      ++ACFUnkInstTranslations;
      offsets[v] = make_shared<UnknownOffsetVal>(v);
      return offsets[v];
    }
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(BinaryOperator *bo) {
    // Generate an operator as a function of the binary operator
    ++ACFBinOpTranslations;

    OffsetOperator op = fromBinaryOpcode(bo->getOpcode());
    if(op == OffsetOperator::end) {
      // We don't handle this kind of operation
      offsets[bo] = make_shared<InstOffsetVal>(bo);
      return offsets[bo];
    }

    OffsetValPtr lhs = getOrCreateVal(bo->getOperand(0));
    OffsetValPtr rhs = getOrCreateVal(bo->getOperand(1));
    offsets[bo] = make_shared<BinOpOffsetVal>(lhs, op, rhs);
    return offsets[bo];
  }

  OffsetValPtr OffsetPropagation::getGEPExpr(Value *ptr, Type *ptr_t, Use *idx_begin, Use *idx_end) {
    ++ACFGEPTranslations;

    const DataLayout& DL = M->getDataLayout();
    // We begin with the offset equal to the base
    OffsetValPtr offset = getOrCreateVal(ptr);

    // Start calculating offsets
    Type *t = ptr_t;

    for(auto i=idx_begin,e=idx_end; i!=e; ++i) {
      OffsetValPtr idx_off;

      if(auto struct_t=dyn_cast<StructType>(t)) {
        // Calculate the offset to the struct element
        OffsetValPtr idx = getOrCreateVal(*i);
        if(!idx->isConst()) {
          return make_shared<UnknownOffsetVal>(ptr);
        }
        assert(idx->isConst()); // Struct references can't be dynamic

        int index = idx->constVal().getZExtValue();
        assert(struct_t->getNumElements() > index);

        // Update the type for next iteration
        t = struct_t->getElementType(index);

        // Calculate the offset for this index
        int elem_off = 0;
        for(int i=0; i<index; ++i) {
          elem_off += DL.getTypeAllocSize(struct_t->getElementType(i));
        }

        // Our element starts at the end of the previous ones
        idx_off = make_shared<ConstOffsetVal>(elem_off);

      } else if(auto seq_t=dyn_cast<SequentialType>(t)) {
        // Calculate the offset to the array element
        OffsetValPtr idx = getOrCreateVal(*i);

        // Calculate the size to step
        OffsetValPtr size = make_shared<ConstOffsetVal>(DL.getTypeAllocSize(seq_t->getElementType()));

        idx_off = make_shared<BinOpOffsetVal>(idx, Mul, size);

        // Update the type for next iteration
        t = seq_t->getElementType();

      } else if(auto seq_t=dyn_cast<PointerType>(t)) {
        // Calculate the offset to the array element
        OffsetValPtr idx = getOrCreateVal(*i);

        // Calculate the size to step
        OffsetValPtr size = make_shared<ConstOffsetVal>(DL.getTypeAllocSize(seq_t->getElementType()));

        idx_off = make_shared<BinOpOffsetVal>(idx, Mul, size);

        // Update the type for next iteration
        t = seq_t->getElementType();

      } else {
        errs() << *t << "\n" << (isa<PointerType>(t) ? "T" : "F") << (isa<SequentialType>(t) ? "T" : "F") << "\n";
        assert(false && "GEP must index a struct or sequence");
      }
      offset = make_shared<BinOpOffsetVal>(offset, Add, idx_off);
    }
    return offset;

  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(GetElementPtrInst *gep) {
    Value *ptr = gep->getPointerOperand();
    Type *ptr_t = gep->getPointerOperandType();
    Use *idx_begin = gep->idx_begin();
    Use *idx_end = gep->idx_end();
    return getGEPExpr(ptr, ptr_t, idx_begin, idx_end);
  }

  OffsetValPtr OffsetPropagation::getOrCreateGEPVal(ConstantExpr *gep) {
    Value *ptr = gep->getOperand(0);
    Type *ptr_t = ptr->getType();
    Use *idx_begin = gep->op_begin()+1;
    Use *idx_end = gep->op_end();
    return getGEPExpr(ptr, ptr_t, idx_begin, idx_end);
  }


  OffsetValPtr OffsetPropagation::getOrCreateVal(CallInst *ci) {
    //TODO
    ++ACFCallTranslations;
    offsets[ci] = make_shared<InstOffsetVal>(ci);
    return offsets[ci];
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(CastInst *ci) {
    // Just drop through the cast for now
    ++ACFCastTranslations;
    return getOrCreateVal(ci->getOperand(0));
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(CmpInst *ci) {
    ++ACFCmpTranslations;
    OffsetValPtr lhs = getOrCreateVal(ci->getOperand(0));
    OffsetValPtr rhs = getOrCreateVal(ci->getOperand(1));
    OffsetOperator op = fromCmpPredicate(ci->getPredicate());
    if(op != OffsetOperator::end)
      offsets[ci] = make_shared<BinOpOffsetVal>(lhs, op, rhs);
    else
      offsets[ci] = make_shared<InstOffsetVal>(ci);
    return offsets[ci];
  }
  bool OffsetPropagation::isUpdateStore(StoreInst *s) {
    vector<pair<int,Value*>> ops;
    ops.push_back(make_pair(0,s));
    while(ops.size()>0) {
      int depth = ops.back().first + 1;
      Value* v = ops.back().second;
      ops.pop_back();

      // We found a load from the same address
      if(auto l=dyn_cast<LoadInst>(v)) {
        if(l->getPointerOperand() == s->getPointerOperand())
          return true;
      }

      // Keep traversing
      if(auto u=dyn_cast<User>(v)) {
        if(depth < 4) {
          for(auto op=u->op_begin(),e=u->op_end(); op!=e; ++op) {
            Value *opv = *op;
            ops.push_back(make_pair(depth,opv));
          }
        }
      }
    }
    return false;
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(Constant *c) {
    //errs() << "Constant: " << *c << "\n";
    //errs() << "Constant Type: " << *c->getType() << "\n";
    if(c->getType()->isIntegerTy() || c->getType()->isPointerTy())
      offsets[c] = make_shared<ConstOffsetVal>(c);
    else
      offsets[c] = make_shared<UnknownOffsetVal>(c);
    return offsets[c];
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(LoadInst *l) {
    ++ACFLoadTranslations;
    Function &f = *l->getParent()->getParent();

    MemoryDependenceResults& MD = getAnalysis<MemoryDependenceWrapperPass>(f).getMemDep();
    MemDepResult res = MD.getDependency(l);

    // Store was found through dependence analysis
    if(res.isDef()) {
      if(auto s=dyn_cast<StoreInst>(res.getInst())) {
        offsets[l]=getOrCreateVal(s->getValueOperand());
        return offsets[l];
      }
    }
    // Attempt manual discovery
    Value *ptr = l->getPointerOperand();
    const DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>(f).getDomTree();
    const PostDominatorTree& PDT = getAnalysis<PostDominatorTreeWrapperPass>(f).getPostDomTree();
    for(auto u=ptr->user_begin(),e=ptr->user_end(); u!=e; ++u) {
      //errs() << "Pointer used in: " << **u << "\n";
      if(auto s=dyn_cast<StoreInst>(*u)) {
        if(s->getPointerOperand() == ptr
                && !PDT.dominates(s->getParent(),l->getParent())
                && !isUpdateStore(s)) {
          // errs() << "Manual Load-Store Pair: \n" << *l << "\n" << *s << "\n";
          offsets[l] = getOrCreateVal(s->getValueOperand());
          return offsets[l];
        }
      }
    }
    // Default, unknown def
    // errs() << "No pair found for load: "<< *l->getPointerOperand() << "\n" << l << " - " << *l << "\n";
    offsets[l] = make_shared<InstOffsetVal>(l);
    return offsets[l];
  }

  OffsetValPtr OffsetPropagation::getOrCreateVal(PHINode *p) {
    ++ACFPhiTranslations;
    // Get the required analysis
    Function &f = *(p->getFunction());
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(f).getDomTree();
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(f).getLoopInfo();

    // Get all incoming values
    std::vector<Value *> fwd_values, bk_values;
    std::vector<BasicBlock *> fwd_blocks, bk_blocks;

    for(int i=0; i<p->getNumIncomingValues(); i++) {
      // Sort incoming values into forward and back
      bool loopedge = isPotentiallyReachable(p->getParent(), p->getIncomingBlock(i), &DT, &LI);

      if(loopedge) {
        bk_values.push_back(p->getIncomingValue(i));
        bk_blocks.push_back(p->getIncomingBlock(i));
        //errs() << "Found looped value: " << *p->getIncomingValue(i) << "\n";
      } else {
        fwd_values.push_back(p->getIncomingValue(i));
        fwd_blocks.push_back(p->getIncomingBlock(i));
      }
    }

    if(fwd_values.size() == 0) {
      offsets[p] = make_shared<InstOffsetVal>(p);
      return offsets[p];
    }

    // Calculate ourselves from non-loops
    offsets[p] = applyDominatingCondition(fwd_values, fwd_blocks, p, DT);

    // TODO: In many cases we can capture loop constructs
    return offsets[p];
  }

  OffsetValPtr OffsetPropagation::applyDominatingCondition(
      std::vector<Value *>& values,
      std::vector<BasicBlock *>& blocks,
      Instruction * mergePt,
      DominatorTree& DT) {

    assert(values.size() == blocks.size());
    assert(values.size() > 0);
    // Base Case
    if(values.size() == 1)
      return getOrCreateVal(values[0]);

    /*
    errs() << "Finding determining condition for values:\n";
    for(auto v=values.begin(),e=values.end(); v!=e; ++v)
      errs() << "\t" << **v << "\n";
    */

    // Locate the common dominator
    BasicBlock *dom = nullptr;
    for(auto b=blocks.begin(), e=blocks.end(); b != e; ++b) {

      if(dom == nullptr)
        dom = *b;
      else
        dom = DT.findNearestCommonDominator(dom, *b);
    }
    assert(dom != nullptr);

    //errs() << "Dominating Block:\n";
    //errs() << "\t" << *dom << "\n";

    // Locate the branch exiting the dominator
    BranchInst *b = dyn_cast<BranchInst>(&*dom->rbegin());
    assert(b != nullptr && b->isConditional());

    OffsetValPtr cond = getOrCreateVal(b->getCondition());
    OffsetValPtr ncond = negateCondition(cond);
    BasicBlock *taken = b->getSuccessor(0);
    BasicBlock *untaken = b->getSuccessor(1);

    /*
    errs() << "Dominating Condition:\n";
      errs() << "\t" << *b << "\n";
    errs() << "Taken Block:\n";
      errs() << "\t" << *taken << "\n";
    errs() << "Untaken Block:\n";
      errs() << "\t" << *untaken << "\n";
    */

    // Calculate values for recursion
    std::vector<Value *> v_taken;
    std::vector<Value *> v_untaken;
    std::vector<BasicBlock *> b_taken;
    std::vector<BasicBlock *> b_untaken;

    // Select for any non-dominating definitions
    for(int i=0; i<values.size(); i++) {
      if(blocks[i] != dom) {
        if(isPotentiallyReachable(taken, blocks[i], &DT, nullptr)) {
          v_taken.push_back(values[i]);
          b_taken.push_back(blocks[i]);
        } else {
          v_untaken.push_back(values[i]);
          b_untaken.push_back(blocks[i]);
        }
      }
    }

    // Insert this definition if this block itself defines the only condition
    for(int i=0; i<values.size(); i++) {
      if(blocks[i] == dom) {
        if(v_taken.size() == 0) {
          v_taken.push_back(values[i]);
          b_taken.push_back(blocks[i]);
        } else {
          v_untaken.push_back(values[i]);
          b_untaken.push_back(blocks[i]);
        }
      }
    }

    // Dirty, dirty hack
    if(v_untaken.size() == 0 && v_taken.size() > 1) {
        v_untaken.push_back(v_taken.back());
        b_untaken.push_back(b_taken.back());
        v_taken.pop_back();
        b_taken.pop_back();
    }

    assert(v_taken.size() > 0);
    assert(v_untaken.size() > 0);
    OffsetValPtr off_taken = applyDominatingCondition(v_taken, b_taken, mergePt, DT);
    OffsetValPtr off_untaken = applyDominatingCondition(v_untaken, b_untaken, mergePt, DT);

    //returning (c * off_taken) + (!c * off_untaken)
    OffsetValPtr mult_taken = make_shared<BinOpOffsetVal>(cond, Mul, off_taken);
    OffsetValPtr mult_untaken = make_shared<BinOpOffsetVal>(ncond, Mul, off_untaken);
    return make_shared<BinOpOffsetVal>(mult_taken, Add, mult_untaken);
  }

  OffsetValPtr OffsetPropagation::inCallContext(const OffsetValPtr& orig, const CallInst *ci) {
    unordered_map<OffsetValPtr, OffsetValPtr> rep;
    const Function *f = ci->getCalledFunction();
    if (f == nullptr)
      return orig; // Can't map into function

    // Build the map from formals to actuals
    auto f_arg = f->arg_begin();
    auto c_arg = ci->arg_begin();
    while(c_arg != ci->arg_end()) {
      rep[make_shared<ArgOffsetVal>(const_cast<Argument *>(&*f_arg))] = getOrCreateVal(*c_arg);
      ++f_arg;
      ++c_arg;
    }

    return replaceComponents(orig, rep);
  }

  OffsetValPtr OffsetPropagation::inGridContext(const OffsetValPtr& orig, int thread_dimx, int thread_dimy, int thread_dimz, int block_dimx, int block_dimy, int block_dimz) {

    if(auto i_off = dyn_cast<InstOffsetVal>(&*orig)) {
      if(auto ci=dyn_cast<CallInst>(i_off->inst)) {
        Function *f = ci->getCalledFunction();
        if(f != nullptr) {
          switch(f->getIntrinsicID()) {
            case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
              return make_shared<ConstOffsetVal>(thread_dimx);
            case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
              return make_shared<ConstOffsetVal>(thread_dimy);
            case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
              return make_shared<ConstOffsetVal>(thread_dimz);
            case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
              return make_shared<ConstOffsetVal>(block_dimx);
            case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
              return make_shared<ConstOffsetVal>(block_dimy);
            case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
              return make_shared<ConstOffsetVal>(block_dimz);
            default:
              break;
          }
        }
      }
    }

    // Recursive case
    auto bo = dyn_cast<BinOpOffsetVal>(&*orig);
    if(!bo)
      return orig; // This is a leaf node that didn't match

    OffsetValPtr lhs = inGridContext(bo->lhs, thread_dimx, thread_dimy, thread_dimz, block_dimx, block_dimy, block_dimz);

    OffsetValPtr rhs = inGridContext(bo->rhs, thread_dimx, thread_dimy, thread_dimz, block_dimx, block_dimy, block_dimz);

    // Attempt to avoid re-allocation if possible
    if(lhs == bo->lhs && rhs == bo->rhs)
      return orig; // No changes were made
    else
      return make_shared<BinOpOffsetVal>(lhs, bo->op, rhs);
  }

  OffsetValPtr OffsetPropagation::inThreadContext(const OffsetValPtr& orig, int thread_idx, int thread_idy, int thread_idz, int block_idx, int block_idy, int block_idz) {

    if(auto i_off = dyn_cast<InstOffsetVal>(&*orig)) {
      if(auto ci=dyn_cast<CallInst>(i_off->inst)) {
        Function *f = ci->getCalledFunction();
        if(f != nullptr) {
          switch(f->getIntrinsicID()) {
            case Intrinsic::nvvm_read_ptx_sreg_tid_x:
              return make_shared<ConstOffsetVal>(thread_idx);
            case Intrinsic::nvvm_read_ptx_sreg_tid_y:
              return make_shared<ConstOffsetVal>(thread_idy);
            case Intrinsic::nvvm_read_ptx_sreg_tid_z:
              return make_shared<ConstOffsetVal>(thread_idz);
            case Intrinsic::nvvm_read_ptx_sreg_laneid:
              return make_shared<ConstOffsetVal>(thread_idx % 32);
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
              return make_shared<ConstOffsetVal>(block_idx);
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
              return make_shared<ConstOffsetVal>(block_idy);
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
              return make_shared<ConstOffsetVal>(block_idz);
            default:
              break;
          }
        }
      }
    }

    // Recursive case
    auto bo = dyn_cast<BinOpOffsetVal>(&*orig);
    if(!bo)
      return orig; // This is a leaf node that didn't match

    OffsetValPtr lhs = inThreadContext(bo->lhs, thread_idx, thread_idy, thread_idz, block_idx, block_idy, block_idz);

    OffsetValPtr rhs = inThreadContext(bo->rhs, thread_idx, thread_idy, thread_idz, block_idx, block_idy, block_idz);

    // Attempt to avoid re-allocation if possible
    if(lhs == bo->lhs && rhs == bo->rhs)
      return orig; // No changes were made
    else
      return make_shared<BinOpOffsetVal>(lhs, bo->op, rhs);
  }

  OffsetOperator OffsetPropagation::fromBinaryOpcode(llvm::Instruction::BinaryOps op) {
    switch(op) {
      case BinaryOperator::Add: return OffsetOperator::Add;
      case BinaryOperator::Sub: return OffsetOperator::Sub;
      case BinaryOperator::Mul: return OffsetOperator::Mul;
      case BinaryOperator::SDiv: return OffsetOperator::SDiv;
      case BinaryOperator::UDiv: return OffsetOperator::UDiv;
      case BinaryOperator::SRem: return OffsetOperator::SRem;
      case BinaryOperator::URem: return OffsetOperator::URem;
      case BinaryOperator::And: return OffsetOperator::And;
      case BinaryOperator::Or: return OffsetOperator::Or;
      case BinaryOperator::Xor: return OffsetOperator::Xor;
      default: return OffsetOperator::end;
    }
  }

  OffsetOperator OffsetPropagation::fromCmpPredicate(llvm::CmpInst::Predicate p) {
    switch(p) {
      case llvm::CmpInst::Predicate::ICMP_EQ: return OffsetOperator::Eq;
      case llvm::CmpInst::Predicate::ICMP_NE: return OffsetOperator::Neq;
      case llvm::CmpInst::Predicate::ICMP_SLT: return OffsetOperator::SLT;
      case llvm::CmpInst::Predicate::ICMP_SLE: return OffsetOperator::SLE;
      case llvm::CmpInst::Predicate::ICMP_SGT: return OffsetOperator::SGT;
      case llvm::CmpInst::Predicate::ICMP_SGE: return OffsetOperator::SGE;
      case llvm::CmpInst::Predicate::ICMP_ULT: return OffsetOperator::ULT;
      case llvm::CmpInst::Predicate::ICMP_ULE: return OffsetOperator::ULE;
      case llvm::CmpInst::Predicate::ICMP_UGT: return OffsetOperator::UGT;
      case llvm::CmpInst::Predicate::ICMP_UGE: return OffsetOperator::UGE;
      default: return OffsetOperator::end;
    }
  }

  std::vector<const Function*> OffsetPropagation::findRequiredContexts(const OffsetValPtr& ptr,
      std::vector<const Function*> found) {
    if(auto bo=dyn_cast<BinOpOffsetVal>(&*ptr)) {
      findRequiredContexts(bo->lhs, found);
      findRequiredContexts(bo->rhs, found);
    }
    if(auto arg=dyn_cast<ArgOffsetVal>(&*ptr)) {
      if(find(found.begin(), found.end(), arg->arg->getParent()) == found.end())
        found.push_back(arg->arg->getParent());
    }
    return found;

  }

  vector<const CallInst*> OffsetPropagation::getSameModuleFunctionCallers(const Function *f) {
    vector<const CallInst*> ret;
    for(auto g=f->getParent()->begin(),e=f->getParent()->end(); g != e; ++g) {
      if(auto f=dyn_cast<Function>(g)) {
        for(auto b=f->begin(),e=f->end(); b!=e; ++b) {
          for(auto i=b->begin(),e=b->end(); i!=e; ++i) {
            if(auto* ci=dyn_cast<CallInst>(i)) {
              if(ci->getCalledFunction() == f)
                ret.push_back(ci);
            }
          }
        }
      }
    }
    return ret;
  }
  vector<OffsetValPtr> OffsetPropagation::inContexts(OffsetValPtr& orig) {
    vector<const Function*> empty;
    vector<OffsetValPtr> iacf = inContexts(orig, empty);

    if(MaxIACFSize.getValue() < iacf.size())
        MaxIACFSize=iacf.size();
    return iacf;
  }

  vector<OffsetValPtr> OffsetPropagation::inContexts(OffsetValPtr& orig, std::vector<const Function *>& ignore) {
    assert(orig != nullptr);
    vector<const Function *> context = findRequiredContexts(orig);
    vector<OffsetValPtr> ret;

    for(auto f=context.begin(),e=context.end(); f!=e; ++f) {
      vector<const CallInst*> callers = getSameModuleFunctionCallers(*f);
      if(callers.empty())
        continue;

      for(auto ci=callers.begin(),e=callers.end(); ci!=e; ++ci) {
        OffsetValPtr inContext = inCallContext(orig, *ci);
        vector<const Function*> recIgnore = ignore;
        recIgnore.push_back(*f); // Don't allow this function's context to be added again
        vector<OffsetValPtr> recurse = inContexts(inContext, recIgnore);

        // Append the recursed results into my results
        for(auto b=recurse.begin(),e=recurse.end(); b!=e; ++b)
          ret.push_back(*b);
      }

      // Done
      return ret;
    }

    //Fall-through, no additional context can be added
    ret.push_back(orig);
    return ret;
  }
}

char gpucheck::OffsetPropagation::ID = 0;
static RegisterPass<gpucheck::OffsetPropagation> X("offset-prop", "Propagates offsets through expressions",
                                        false,
                                        true);
#undef DEBUG_TYPE
