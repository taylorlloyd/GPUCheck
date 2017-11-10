#include "ThreadDepAnalysis.h"
#include "OffsetOps.h"

using namespace llvm;
using namespace std;

namespace gpucheck {

  OffsetValPtr negateCondition(OffsetValPtr& cond) {
    assert(isa<BinOpOffsetVal>(cond.get()));
    auto b=dyn_cast<BinOpOffsetVal>(cond.get());
    OffsetValPtr lhs = b->lhs;
    OffsetValPtr rhs = b->rhs;
    // Logical conditionals, apply DeMorgan's laws.
    if (b->op == OffsetOperator::And)
      return make_shared<BinOpOffsetVal>(negateCondition(lhs), OffsetOperator::Or, negateCondition(rhs));
    else if (b->op == OffsetOperator::Or)
      return make_shared<BinOpOffsetVal>(negateCondition(lhs), OffsetOperator::And, negateCondition(rhs));
    // Comparison conditions are negated by flipping the operator.
    OffsetOperator flipped;
    switch(b->op) {
      case OffsetOperator::Eq: flipped = Neq; break;
      case OffsetOperator::Neq: flipped = Eq; break;
      case OffsetOperator::SLT: flipped = SGE; break;
      case OffsetOperator::SGE: flipped = SLT; break;
      case OffsetOperator::SLE: flipped = SGT; break;
      case OffsetOperator::SGT: flipped = SLE; break;
      case OffsetOperator::ULT: flipped = UGE; break;
      case OffsetOperator::UGE: flipped = ULT; break;
      case OffsetOperator::ULE: flipped = UGT; break;
      case OffsetOperator::UGT: flipped = ULE; break;
      default: flipped = end; break;
    }
    assert(flipped != OffsetOperator::end);
    return make_shared<BinOpOffsetVal>(b->lhs, flipped, b->rhs);
  }

  OffsetValPtr sumOfProducts(OffsetValPtr ov) {    
    OffsetValPtr tmp = ov;
    OffsetValPtr res = sumOfProductsPass(ov);
    while (!matchingOffsets(tmp, res)) {
      tmp = res;
      res = sumOfProductsPass(tmp);
    }
    return res;
  }

  OffsetValPtr sumOfProductsPass(OffsetValPtr ov) {
    auto bo=dyn_cast<BinOpOffsetVal>(&*ov);
    if(bo == nullptr)
      return ov;

    // We're working with a binary operator
    OffsetValPtr lhs = sumOfProductsPass(bo->lhs);
    OffsetValPtr rhs = sumOfProductsPass(bo->rhs);

    if(bo->op == OffsetOperator::Mul) {
      auto lhs_bo=dyn_cast<BinOpOffsetVal>(&*lhs);
      if(lhs_bo != nullptr && (lhs_bo->op == OffsetOperator::Add || lhs_bo->op == OffsetOperator::Sub)) {
        // Multiply RHS into LHS operands
        auto new_lhs = make_shared<BinOpOffsetVal>(lhs_bo->lhs, bo->op, rhs);
        auto new_rhs = make_shared<BinOpOffsetVal>(lhs_bo->rhs, bo->op, rhs);
        return make_shared<BinOpOffsetVal>(new_lhs, lhs_bo->op, new_rhs);
      }

      auto rhs_bo=dyn_cast<BinOpOffsetVal>(&*rhs);
      if(rhs_bo != nullptr && (rhs_bo->op == OffsetOperator::Add || rhs_bo->op == OffsetOperator::Sub)) {
        // Multiply LHS into RHS operands
        auto new_lhs = make_shared<BinOpOffsetVal>(lhs, bo->op, rhs_bo->lhs);
        auto new_rhs = make_shared<BinOpOffsetVal>(lhs, bo->op, rhs_bo->rhs);
        return make_shared<BinOpOffsetVal>(new_lhs, rhs_bo->op, new_rhs);
      }
    }
    else if (bo->op == OffsetOperator::SDiv || bo->op == OffsetOperator::UDiv) {
      auto lhs_bo=dyn_cast<BinOpOffsetVal>(&*lhs);
      if(lhs_bo != nullptr && (lhs_bo->op == OffsetOperator::Add || lhs_bo->op == OffsetOperator::Sub)) {
        auto new_lhs = make_shared<BinOpOffsetVal>(lhs_bo->lhs, bo->op, rhs);
        auto new_rhs = make_shared<BinOpOffsetVal>(lhs_bo->rhs, bo->op, rhs);
        return make_shared<BinOpOffsetVal>(new_lhs, lhs_bo->op, new_rhs);
      }
    }

    // Just return the sum-of-productsed operands
    return make_shared<BinOpOffsetVal>(lhs, bo->op, rhs);
  }

  OffsetValPtr simplifyConditions(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs) {
    // Convert (cond_1 - cond_2) to (cond_1*(!cond_2))
    if(auto bo_lhs=dyn_cast<BinOpOffsetVal>(&*lhs)) {
      if(auto bo_rhs=dyn_cast<BinOpOffsetVal>(&*rhs)) {
        if(bo_lhs->isCompare() && bo_rhs->isCompare() && op == OffsetOperator::Sub) {
          return make_shared<BinOpOffsetVal>(lhs, OffsetOperator::Mul, negateCondition(rhs));
        }
      }
    }
    return nullptr;
  }

  OffsetValPtr simplifyConstantVal(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs) {
    assert(lhs->isConst() && rhs->isConst());
    APInt lhsi = lhs->constVal();
    APInt rhsi = rhs->constVal();

    // Always just work in the larger bitwidth
    if(lhsi.getBitWidth() > rhsi.getBitWidth())
      rhsi = rhsi.zext(lhsi.getBitWidth());

    if(rhsi.getBitWidth() > lhsi.getBitWidth())
      lhsi = lhsi.zext(rhsi.getBitWidth());

    APInt out;
    switch(op) {
      case OffsetOperator::Add: out = lhsi + rhsi; break;
      case OffsetOperator::Sub: out = lhsi - rhsi; break;
      case OffsetOperator::Mul: out = lhsi * rhsi; break;
      case OffsetOperator::SDiv: out = lhsi.sdiv(rhsi); break;
      case OffsetOperator::UDiv: out = lhsi.udiv(rhsi); break;
      case OffsetOperator::SRem: out = lhsi.srem(rhsi); break;
      case OffsetOperator::URem: out = lhsi.urem(rhsi); break;
      //case OffsetOperator::And: out = lhsi.And(rhsi); break;
      //case OffsetOperator::Or: out = lhsi.Or(rhsi); break;
      //case OffsetOperator::Xor: out = lhsi.Xor(rhsi); break;
      case OffsetOperator::Eq: out = lhsi.eq(rhsi); break;
      case OffsetOperator::Neq: out = lhsi.ne(rhsi); break;
      case OffsetOperator::SLT: out = lhsi.slt(rhsi); break;
      case OffsetOperator::SLE: out = lhsi.sle(rhsi); break;
      case OffsetOperator::ULT: out = lhsi.ult(rhsi); break;
      case OffsetOperator::ULE: out = lhsi.ule(rhsi); break;
      case OffsetOperator::SGT: out = lhsi.sgt(rhsi); break;
      case OffsetOperator::SGE: out = lhsi.sge(rhsi); break;
      case OffsetOperator::UGT: out = lhsi.ugt(rhsi); break;
      case OffsetOperator::UGE: out = lhsi.uge(rhsi); break;
      case OffsetOperator::end: assert(false); break;
    }
    return make_shared<ConstOffsetVal>(out);
  }

  OffsetValPtr simplifyOffsetVal(OffsetValPtr ov) {
    auto bo=dyn_cast<BinOpOffsetVal>(&*ov);
    if(bo == nullptr)
      return ov;

    OffsetValPtr lhs = simplifyOffsetVal(bo->lhs);
    OffsetValPtr rhs = simplifyOffsetVal(bo->rhs);
    if(lhs->isConst() && rhs->isConst())
      return simplifyConstantVal(lhs, bo->op, rhs);

    switch(bo->op) {
      case OffsetOperator::Add:
      {
        // Adding zero does nothing
        if(rhs->isConst() && rhs->constVal() == 0)
          return lhs;
        if(lhs->isConst() && lhs->constVal() == 0)
          return rhs;
      }
      case OffsetOperator::Sub:
      {
        if(rhs->isConst() && rhs->constVal() == 0)
          return lhs;
        if(auto new_bo = simplifyConditions(lhs, bo->op, rhs))
          return simplifyOffsetVal(new_bo);

      }
      case OffsetOperator::Mul:
      {
        // Zeroes destroy the entire tree
        if(rhs->isConst() && rhs->constVal() == 0)
          return rhs;
        if(lhs->isConst() && lhs->constVal() == 0)
          return lhs;
        // Ones have no effect
        if(rhs->isConst() && rhs->constVal() == 1)
          return lhs;
        if(lhs->isConst() && lhs->constVal() == 1)
          return rhs;
      }
      case OffsetOperator::SDiv:
      case OffsetOperator::UDiv:
      {
        // Dividing by one does nothing
        if(rhs->isConst() && rhs->constVal() == 1)
          return lhs;
        // 0/anything is zero
        if(lhs->isConst() && lhs->constVal() == 0)
          return lhs;
      }
      case OffsetOperator::SRem:
      case OffsetOperator::URem:
      {
        // 0%anything is always 0
        if (lhs->isConst() && lhs->constVal() == 0)
          return lhs;
        // 1%anything is always 1
        if (lhs->isConst() && lhs->constVal() == 1)
          return lhs;
        // anything%1 is always 0
        if (rhs->isConst() && rhs->constVal() == 1)
          return make_shared<ConstOffsetVal>(0);
      }
    }

    if (OffsetValPtr simp = simplifyConstantSubExpressions(lhs, bo->op, rhs)) {
      return simp;
    }

    // Just return the simplified components
    return make_shared<BinOpOffsetVal>(lhs, bo->op, rhs);
  }

  OffsetValPtr simplifyConstantSubExpressions(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs)
  {
    bool boAdd = (op == OffsetOperator::Add);
    bool boSub = (op == OffsetOperator::Sub);

    if (isa<BinOpOffsetVal>(&*lhs) && rhs->isConst() && (boAdd || boSub)) {
      auto lhsBinop = dyn_cast<BinOpOffsetVal>(&*lhs);
      OffsetValPtr llhs = lhsBinop->lhs;
      OffsetValPtr lrhs = lhsBinop->rhs;
      if (lrhs->isConst()) {
        switch (lhsBinop->op) {
        case OffsetOperator::Add: {
          APInt newConst = boAdd ? lrhs->constVal() + rhs->constVal() : lrhs->constVal() - rhs->constVal();
          OffsetValPtr result = make_shared<BinOpOffsetVal>(llhs, lhsBinop->op, make_shared<ConstOffsetVal>(newConst));
          return simplifyOffsetVal(result);
        } break;
        case OffsetOperator::Sub: {
          APInt newConst = boAdd ? lrhs->constVal() - rhs->constVal() : lrhs->constVal() + rhs->constVal();
          OffsetValPtr result = make_shared<BinOpOffsetVal>(llhs, lhsBinop->op, make_shared<ConstOffsetVal>(newConst));
          return simplifyOffsetVal(result);
          break;
        }
        }
      }
      else if (llhs->isConst()) {
        switch (lhsBinop->op) {
        case OffsetOperator::Sub:
        case OffsetOperator::Add: {
          APInt newConst = boAdd ? llhs->constVal() + rhs->constVal() : llhs->constVal() - rhs->constVal();
          OffsetValPtr result = make_shared<BinOpOffsetVal>(make_shared<ConstOffsetVal>(newConst), lhsBinop->op, lrhs);
          return simplifyOffsetVal(result);
        } break;
        }
      }
    }
    if (isa<BinOpOffsetVal>(&*rhs) && lhs->isConst() && (boAdd || boSub)) {
      auto rhsBinop = dyn_cast<BinOpOffsetVal>(&*rhs);
      OffsetValPtr rlhs = rhsBinop->lhs;
      OffsetValPtr rrhs = rhsBinop->rhs;
      if (rlhs->isConst()) {
        switch (rhsBinop->op) {
        case OffsetOperator::Add: {
          APInt newConst = boAdd ? lhs->constVal() + rlhs->constVal() : lhs->constVal() - rlhs->constVal();
          OffsetOperator newOp = boAdd ? rhsBinop->op : OffsetOperator::Sub;
          OffsetValPtr result = make_shared<BinOpOffsetVal>(make_shared<ConstOffsetVal>(newConst), newOp, rrhs);
          return simplifyOffsetVal(result);
        } break;
        case OffsetOperator::Sub: {
          APInt newConst = boAdd ? lhs->constVal() + rlhs->constVal() : lhs->constVal() - rlhs->constVal();
          OffsetOperator newOp = boAdd ? rhsBinop->op : OffsetOperator::Add;
          OffsetValPtr result = make_shared<BinOpOffsetVal>(make_shared<ConstOffsetVal>(newConst), newOp, rrhs);
          return simplifyOffsetVal(result);
        } break;
        }
      }
      if (rrhs->isConst()) {
        switch (rhsBinop->op) {
        case OffsetOperator::Add: {
          APInt newConst = boAdd ? lhs->constVal() + rrhs->constVal() : lhs->constVal() - rrhs->constVal();
          OffsetOperator newOp = boAdd ? rhsBinop->op : OffsetOperator::Sub;
          OffsetValPtr result = make_shared<BinOpOffsetVal>(make_shared<ConstOffsetVal>(newConst), newOp, rlhs);
          return simplifyOffsetVal(result);
        } break;
        case OffsetOperator::Sub: {
          APInt newConst = boAdd ? lhs->constVal() - rrhs->constVal() : lhs->constVal() + rrhs->constVal();
          OffsetOperator newOp = boAdd ? rhsBinop->op : OffsetOperator::Sub;
          OffsetValPtr result = make_shared<BinOpOffsetVal>(make_shared<ConstOffsetVal>(newConst), newOp, rlhs);
          return simplifyOffsetVal(result);
        } break;
        }
      }
    }
    return nullptr;
  }

  bool matchingOffsets(OffsetValPtr lhs, OffsetValPtr rhs) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    if(lhs->isConst() && rhs->isConst()) {
      APInt lhs_c = lhs->constVal();
      APInt rhs_c = rhs->constVal();
      uint64_t bitwidth = max(lhs_c.getBitWidth(), rhs_c.getBitWidth());
      return lhs_c.sextOrSelf(bitwidth) == rhs_c.sextOrSelf(bitwidth);
    }

    auto i_lhs = dyn_cast<InstOffsetVal>(&*lhs);
    auto i_rhs = dyn_cast<InstOffsetVal>(&*rhs);
    if(i_lhs && i_rhs) {
      return (i_lhs->inst == i_rhs->inst);
    }

    auto a_lhs = dyn_cast<ArgOffsetVal>(&*lhs);
    auto a_rhs = dyn_cast<ArgOffsetVal>(&*rhs);
    if(a_lhs && a_rhs) {
      return (a_lhs->arg == a_rhs->arg);
    }

    auto u_lhs = dyn_cast<UnknownOffsetVal>(&*lhs);
    auto u_rhs = dyn_cast<UnknownOffsetVal>(&*rhs);
    if(u_lhs && u_rhs) {
      return (u_lhs->cause == u_rhs->cause);
    }

    auto bo_lhs = dyn_cast<BinOpOffsetVal>(&*lhs);
    auto bo_rhs = dyn_cast<BinOpOffsetVal>(&*rhs);
    if(bo_lhs && bo_rhs) {
      return bo_lhs->op == bo_rhs->op
        && matchingOffsets(bo_lhs->lhs, bo_rhs->lhs)
        && matchingOffsets(bo_lhs->rhs, bo_rhs->rhs);
    }

    return false;
  }

  bool equalOffsets(OffsetValPtr lhs, OffsetValPtr rhs, ThreadDependence& td) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    if(lhs->isConst() && rhs->isConst()) {
      APInt lhs_c = lhs->constVal();
      APInt rhs_c = rhs->constVal();
      uint64_t bitwidth = max(lhs_c.getBitWidth(), rhs_c.getBitWidth());
      return lhs_c.sextOrSelf(bitwidth) == rhs_c.sextOrSelf(bitwidth);
    }

    auto i_lhs = dyn_cast<InstOffsetVal>(&*lhs);
    auto i_rhs = dyn_cast<InstOffsetVal>(&*rhs);
    if(i_lhs && i_rhs) {
      if(i_lhs->inst == i_rhs->inst)
        return !td.isDependent(const_cast<Instruction *>(i_lhs->inst));
    }

    auto a_lhs = dyn_cast<ArgOffsetVal>(&*lhs);
    auto a_rhs = dyn_cast<ArgOffsetVal>(&*rhs);
    if(a_lhs && a_rhs) {
      if(a_lhs->arg == a_rhs->arg)
        return !td.isDependent(const_cast<Argument *>(a_lhs->arg));
    }

    auto u_lhs = dyn_cast<UnknownOffsetVal>(&*lhs);
    auto u_rhs = dyn_cast<UnknownOffsetVal>(&*rhs);
    if(u_lhs && u_rhs) {
      if(u_lhs->cause == u_rhs->cause)
        return !td.isDependent(const_cast<Value *>(u_lhs->cause));
    }

    auto bo_lhs = dyn_cast<BinOpOffsetVal>(&*lhs);
    auto bo_rhs = dyn_cast<BinOpOffsetVal>(&*rhs);
    if(bo_lhs && bo_rhs) {
      return bo_lhs->op == bo_rhs->op
        && equalOffsets(bo_lhs->lhs, bo_rhs->lhs, td)
        && equalOffsets(bo_lhs->rhs, bo_rhs->rhs, td);
    }

    return false;
  }

  void addToVector(const OffsetValPtr& ov, vector<OffsetValPtr>& add, vector<OffsetValPtr>&sub, bool isSub = false) {
    assert(ov != nullptr);
    auto bo = dyn_cast<BinOpOffsetVal>(&*ov);
    if(bo != nullptr) {
      if(bo->op == Add) {
        addToVector(bo->lhs, add, sub, isSub);
        addToVector(bo->rhs, add, sub, isSub);
        return;
      }
      if(bo->op == Sub) {
        addToVector(bo->lhs, add, sub, isSub);
        addToVector(bo->rhs, add, sub, !isSub);
        return;
      }
    }
    if(isSub) {
      sub.push_back(ov);
    } else {
      add.push_back(ov);
    }
  }

  OffsetValPtr cancelDiffs(OffsetValPtr ov, ThreadDependence& td) {
    assert(ov != nullptr);
    OffsetValPtr sop = sumOfProducts(ov);
    // Convert from binary tree to n-ary addition and subtraction
    vector<OffsetValPtr> added;
    vector<OffsetValPtr> subtracted;
    added.clear();
    subtracted.clear();
    addToVector(ov, added, subtracted);

    // Cancel any matching trees in the sums
    bool changed = true;
    while(changed) {
      changed = false;
      for(auto o_a=added.begin(),e_a=added.end(); o_a!=e_a; ++o_a) {
        for(auto o_s=subtracted.begin(),e_s=subtracted.end(); o_s!=e_s; ++o_s) {
          assert(*o_s != nullptr);
          if(equalOffsets(*o_a, *o_s, td)) {
            added.erase(o_a);
            subtracted.erase(o_s);
            changed = true;
            break;
          }
          if (OffsetValPtr simp = simplifyDifferenceOfProducts(*o_a, *o_s, td)) {
            added.erase(o_a);
            subtracted.erase(o_s);
            changed = true;
            addToVector(simp, added,subtracted);
            break;
          }
        }
        if(changed)
          break;
      }
    }

    // Rebuild the binary tree
    OffsetValPtr ret = (added.size() == 0) ? make_shared<ConstOffsetVal>(0) : added.back();
    if(added.size() > 0)
      added.pop_back();

    while(added.size() > 0) {
      ret = make_shared<BinOpOffsetVal>(ret, Add, added.back());
      added.pop_back();
    }

    while(subtracted.size() > 0) {
      ret = make_shared<BinOpOffsetVal>(ret, Sub, subtracted.back());
      subtracted.pop_back();
    }
    return simplifyOffsetVal(ret);
  }

  OffsetValPtr replaceComponents(const OffsetValPtr& orig, std::unordered_map<OffsetValPtr, OffsetValPtr>& rep) {
    // Our loose definition of equality is a problem here
    for(auto r=rep.begin(),e=rep.end(); r!=e; ++r) {
      // Perform a tree-match
      if(matchingOffsets(orig, r->first))
        return r->second;
    }

    auto bo = dyn_cast<BinOpOffsetVal>(&*orig);
    if(!bo)
      return orig; // This is a leaf node that didn't match

    OffsetValPtr lhs = replaceComponents(bo->lhs, rep);
    OffsetValPtr rhs = replaceComponents(bo->rhs, rep);

    // Attempt to avoid re-allocation if possible
    if(lhs == bo->lhs && rhs == bo->rhs)
      return orig; // No changes were made
    else
      return make_shared<BinOpOffsetVal>(lhs, bo->op, rhs);
  }

  // Returns NULL if unable to change anything
  OffsetValPtr simplifyDifferenceOfProducts(OffsetValPtr addt, OffsetValPtr subt, ThreadDependence& td) {
    auto bo_a = dyn_cast<BinOpOffsetVal>(&*addt);
    auto bo_s = dyn_cast<BinOpOffsetVal>(&*subt);
    if (bo_a && bo_s && bo_a->op == OffsetOperator::Mul && bo_s->op == OffsetOperator::Mul) {
      OffsetValPtr a_lhs = bo_a->lhs, s_lhs = bo_s->lhs;
      OffsetValPtr a_rhs = bo_a->rhs, s_rhs = bo_s->rhs;
      if (equalOffsets(a_rhs, s_rhs, td)) {
        // ax-bx
        OffsetValPtr origDiff = make_shared<BinOpOffsetVal>(addt, OffsetOperator::Sub, subt);
        // (a-b)
        OffsetValPtr lhsDiff = make_shared<BinOpOffsetVal>(a_lhs, OffsetOperator::Sub, s_lhs);
        // cancellDiff on (a-b)
        OffsetValPtr new_lhs = cancelDiffs(lhsDiff, td);
        OffsetValPtr new_binop = make_shared<BinOpOffsetVal>(new_lhs, OffsetOperator::Mul, s_rhs);
        OffsetValPtr newsop = sumOfProducts(new_binop);
        OffsetValPtr oldsop = sumOfProducts(origDiff);
        // Did not achieve anything, important for termination.
        if (matchingOffsets(simplifyOffsetVal(newsop), simplifyOffsetVal(oldsop)))
          return nullptr;
        else
          return newsop;
      }
      else if (equalOffsets(a_lhs, s_lhs, td)) {
        OffsetValPtr origDiff = make_shared<BinOpOffsetVal>(addt, OffsetOperator::Sub, subt);
        OffsetValPtr rhsDiff = make_shared<BinOpOffsetVal>(a_rhs, OffsetOperator::Sub, s_rhs);
        OffsetValPtr new_rhs = cancelDiffs(rhsDiff, td);
        OffsetValPtr new_binop = make_shared<BinOpOffsetVal>(s_lhs, OffsetOperator::Mul, new_rhs);
        OffsetValPtr newsop = sumOfProducts(new_binop);
        OffsetValPtr oldsop = sumOfProducts(origDiff);
        if (matchingOffsets(simplifyOffsetVal(newsop), simplifyOffsetVal(oldsop)))
          return nullptr;
        else
          return newsop;
      }
    }
    return nullptr;
  }
}
