#include "OffsetPropagation.h"
#include "OffsetOps.h"
#include "raw_os_ostream.h"

namespace gpucheck {
  const APInt& min(const APInt &lhs, const APInt &rhs) {
    if(lhs.slt(rhs))
      return lhs;
    return rhs;
  }
  const APInt& max(const APInt &lhs, const APInt &rhs) {
    if(lhs.sgt(rhs))
      return lhs;
    return rhs;
  }
  /************************************************
   * OffsetVal
   ************************************************/
  void OffsetVal::print(std::ostream& os) const {
    os << "<ERROR BaseClass print>>";
  }
  const llvm::APInt& OffsetVal::constVal() const {
    assert(false);
  }

  const std::pair<llvm::APInt, llvm::APInt> OffsetVal::constRange() const {
    assert(false);
  }

  bool OffsetVal::isConst() const {
    assert(false);
  }

  /************************************************
   * ConstOffsetVal
   ************************************************/
  void ConstOffsetVal::print(std::ostream& os) const {
    llvm::raw_os_ostream ros(os);
    ros << this->intVal;
  }
  const llvm::APInt& ConstOffsetVal::constVal() const {
    return this->intVal;
  }
  const std::pair<llvm::APInt, llvm::APInt> ConstOffsetVal::constRange() const {
    return make_pair(this->intVal, this->intVal);
  }

  /************************************************
   * InstOffsetVal
   ************************************************/
  void InstOffsetVal::print(std::ostream& os) const {
    llvm::raw_os_ostream ros(os);
    this->inst->printAsOperand(ros, true, this->inst->getModule());
  }
  const llvm::APInt& InstOffsetVal::constVal() const {
    assert(false);
  }
  const std::pair<llvm::APInt, llvm::APInt> InstOffsetVal::constRange() const {
    if(this->inst->getType()->isIntegerTy()) {
      int bitwidth = inst->getType()->getIntegerBitWidth();
      return make_pair(APInt::getSignedMinValue(bitwidth), APInt::getSignedMaxValue(bitwidth));
    }
    return make_pair(APInt::getSignedMinValue(64), APInt::getSignedMaxValue(64));
  }

  /************************************************
   * ArgOffsetVal
   ************************************************/
  void ArgOffsetVal::print(std::ostream& os) const {
    llvm::raw_os_ostream ros(os);
    this->arg->printAsOperand(ros, true, this->arg->getParent()->getParent());
  }
  const llvm::APInt& ArgOffsetVal::constVal() const {
    assert(false);
  }
  const std::pair<llvm::APInt, llvm::APInt> ArgOffsetVal::constRange() const {
    if(this->arg->getType()->isIntegerTy()) {
      int bitwidth = arg->getType()->getIntegerBitWidth();
      return make_pair(APInt::getSignedMinValue(bitwidth), APInt::getSignedMaxValue(bitwidth));
    }
    return make_pair(APInt::getSignedMinValue(64), APInt::getSignedMaxValue(64));
  }

  /************************************************
   * UnknownOffsetVal
   ************************************************/
  void UnknownOffsetVal::print(std::ostream& os) const {
    os << "(unknown on ";
    llvm::raw_os_ostream ros(os);
    this->cause->print(ros, false);
    os << ')';
  }
  const llvm::APInt& UnknownOffsetVal::constVal() const {
    assert(false);
  }
  const std::pair<llvm::APInt, llvm::APInt> UnknownOffsetVal::constRange() const {
    return make_pair(APInt::getSignedMinValue(64), APInt::getSignedMaxValue(64));
  }

  /************************************************
   * BinOpOffsetVal
   ************************************************/
  const std::string BinOpOffsetVal::getPrintOp() const {
    switch(op) {
      case OffsetOperator::Add: return "+";
      case OffsetOperator::Sub: return "-";
      case OffsetOperator::Mul: return "*";
      case OffsetOperator::SDiv:
      case OffsetOperator::UDiv: return "/";
      case OffsetOperator::SRem:
      case OffsetOperator::URem: return "%";
      case OffsetOperator::And: return "&&";
      case OffsetOperator::Or: return "||";
      case OffsetOperator::Xor: return "^";
      case OffsetOperator::Eq: return "==";
      case OffsetOperator::Neq: return "!=";
      case OffsetOperator::SLT:
      case OffsetOperator::ULT: return "<";
      case OffsetOperator::SLE:
      case OffsetOperator::ULE: return "<=";
      case OffsetOperator::SGT:
      case OffsetOperator::UGT: return ">";
      case OffsetOperator::SGE:
      case OffsetOperator::UGE: return ">=";
      case OffsetOperator::end: assert(false);
    }
  }

  void BinOpOffsetVal::print(std::ostream& os) const {
    os << '(' << *lhs << ' ' << getPrintOp() << ' ' << *rhs << ')';
  }
  bool BinOpOffsetVal::isConst() const {
    return false;
  }
  bool BinOpOffsetVal::isCompare() const {
    switch(op) {
      case OffsetOperator::Eq:
      case OffsetOperator::Neq:
      case OffsetOperator::SLT:
      case OffsetOperator::ULT:
      case OffsetOperator::SLE:
      case OffsetOperator::ULE:
      case OffsetOperator::SGT:
      case OffsetOperator::UGT:
      case OffsetOperator::SGE:
      case OffsetOperator::UGE:
        return true;
      default:
        return false;
    }
  }
  const llvm::APInt& BinOpOffsetVal::constVal() const {
    assert(false);
  }
  const std::pair<llvm::APInt, llvm::APInt> BinOpOffsetVal::constRange() const {
    auto lhs_rge = lhs->constRange(), rhs_rge = rhs->constRange();

    // Collect same bitwidths
    uint64_t bitwidth = std::max(
            std::max(lhs_rge.first.getBitWidth(), lhs_rge.second.getBitWidth()),
            std::max(rhs_rge.first.getBitWidth(), rhs_rge.second.getBitWidth()));
    lhs_rge.first = lhs_rge.first.sextOrSelf(bitwidth);
    lhs_rge.second = lhs_rge.second.sextOrSelf(bitwidth);
    rhs_rge.first = rhs_rge.first.sextOrSelf(bitwidth);
    rhs_rge.second = rhs_rge.second.sextOrSelf(bitwidth);

    APInt lower, upper;
    switch(op) {
      case OffsetOperator::Add:
        {
          lower = lhs_rge.first + rhs_rge.first;
          upper = lhs_rge.second + rhs_rge.second;
          break;
        }
      case OffsetOperator::Sub:
        {
          if((lhs_rge.first.isMinSignedValue() && lhs_rge.second.isMaxSignedValue()) &&
             (rhs_rge.first.isMinSignedValue() && rhs_rge.second.isMaxSignedValue())) {
            lower = lhs_rge.first;
            upper = lhs_rge.second;
            break;
          }
          lower = lhs_rge.first - rhs_rge.second;
          upper = lhs_rge.second - rhs_rge.first;
          break;
        }
      case OffsetOperator::Mul:
        {
          // Just calculate the possibilities
          auto a = lhs_rge.first * rhs_rge.first;
          auto b = lhs_rge.first * rhs_rge.second;
          auto c = lhs_rge.second * rhs_rge.first;
          auto d = lhs_rge.second * rhs_rge.second;
          lower = min(min(a,b),min(c,d));
          upper = max(max(a,b),max(c,d));
          break;
        }
      case OffsetOperator::SDiv:
      case OffsetOperator::UDiv:
        {
          if(rhs_rge.first.isNonNegative() && lhs_rge.first.isNonNegative()) {
            lower = lhs_rge.first.sdiv(rhs_rge.second);
            upper = lhs_rge.second.sdiv(rhs_rge.first);
          } else {
            int bitwidth = lhs_rge.first.getBitWidth();
            lower = APInt::getSignedMinValue(bitwidth);
            upper = APInt::getSignedMaxValue(bitwidth);
          }
          break;
        }
      case OffsetOperator::SRem:
      case OffsetOperator::URem:
        {
          if(rhs_rge.first.isNonNegative() && lhs_rge.first.isNonNegative()) {
            lower = APInt(lhs_rge.first.getBitWidth(), 0, false);
            upper = rhs_rge.second;
          } else {
            int bitwidth = lhs_rge.first.getBitWidth();
            lower = APInt::getSignedMinValue(bitwidth);
            upper = APInt::getSignedMaxValue(bitwidth);
          }
          break;
        }
      case OffsetOperator::And:
        {
          // Simplification, but a good boundary
          lower = min(APInt(lhs_rge.first.getBitWidth(), 0, false), lhs_rge.first);
          upper = min(lhs_rge.second, rhs_rge.second);
          break;
        }
      case OffsetOperator::Or:
        {
          lower = min(APInt(lhs_rge.first.getBitWidth(), 0, false), lhs_rge.first);
          upper = max(lhs_rge.second, rhs_rge.second);
          break;
        }
      case OffsetOperator::Xor:
        {
          lower = min(APInt(lhs_rge.first.getBitWidth(), 0, false), lhs_rge.first);
          upper = max(lhs_rge.second, rhs_rge.second);
          break;
        }
      case OffsetOperator::Eq:
      case OffsetOperator::Neq:
      case OffsetOperator::SLT:
      case OffsetOperator::ULT:
      case OffsetOperator::SLE:
      case OffsetOperator::ULE:
      case OffsetOperator::SGT:
      case OffsetOperator::UGT:
      case OffsetOperator::SGE:
      case OffsetOperator::UGE:
        {
          lower = APInt(1, -1, true);
          upper = APInt(1, 0, true);
          break;
        }
      case OffsetOperator::end: assert(false);
    }
    return make_pair(lower, upper);
  }
}
