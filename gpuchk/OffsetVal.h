#include <iostream>
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"

#ifndef OFFSET_VAL_H
#define OFFSET_VAL_H

namespace gpucheck {
  /**
   * Enable any class with method print(ostream) to be used with the streaming API
   */
  template<class T>
  auto operator<<(std::ostream& os, const T& t) -> decltype(t.print(os), os)
  {
    t.print(os);
    return os;
  }

  /**
   * Generic superclass for offsets
   */
  class OffsetVal {
    // dyn_cast support
    public:
      enum OVKind {
        OV_Const,
        OV_Inst,
        OV_Arg,
        OV_BinOp,
        OV_Unk
      };
    private:
      const OVKind kind;
    public:
      OffsetVal(OVKind kind) : kind(kind) {}
      /**
       * Returns true if this OffsetVal contains only a constant part
       */
      virtual bool isConst() const;
      /**
       * Returns an APInt representing this OffsetVal, if it's constant
       */
      virtual const llvm::APInt& constVal() const;
      /**
       * Returns a pair of values from lower to upper, inclusive
       */
      virtual const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      /**
       * Print a human-readable representation of this value
       */
      virtual void print(std::ostream& stream) const;
      OVKind getKind() const { return kind; }
  };

  /**
   * Declare a shorthand pointer type
   */
  typedef std::shared_ptr<OffsetVal> OffsetValPtr;

  /**
   * OffsetVal specialization for constant values
   */
  class ConstOffsetVal : public OffsetVal {
    private:
      const llvm::APInt intVal;
    public:
      ConstOffsetVal(llvm::Constant* c) : OffsetVal(OV_Const), intVal(c->getUniqueInteger()) { }
      ConstOffsetVal(llvm::APInt a) : OffsetVal(OV_Const), intVal(a) { }
      ConstOffsetVal(int i) : OffsetVal(OV_Const), intVal(llvm::APInt(32, i, true)) { }
      bool isConst() const {return true;}
      const llvm::APInt& constVal() const;
      const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      void print(std::ostream& stream) const;

      static bool classof(const OffsetVal *ov) { return ov->getKind() == OV_Const; }
  };

  /**
   * OffsetVal specialization for runtime-known values
   */
  class InstOffsetVal : public OffsetVal {
    public:
      const llvm::Instruction* inst;
      InstOffsetVal(llvm::Instruction* i) : OffsetVal(OV_Inst), inst(i) {
        assert(i != nullptr);
      }
      bool isConst() const {return false;}
      const llvm::APInt& constVal() const;
      const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      void print(std::ostream& stream) const;

      static bool classof(const OffsetVal *ov) { return ov->getKind() == OV_Inst; }
  };

  /**
   * OffsetVal specialization for function parameters
   */
  class ArgOffsetVal : public OffsetVal {
    public:
      const llvm::Argument* arg;
      ArgOffsetVal(llvm::Argument* a) : OffsetVal(OV_Arg), arg(a) {
        assert(a != nullptr);
      }
      bool isConst() const {return false;}
      const llvm::APInt& constVal() const;
      const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      void print(std::ostream& stream) const;

      static bool classof(const OffsetVal *ov) { return ov->getKind() == OV_Arg; }
  };

  /**
   * OffsetVal specialization for unknown values, even at runtime.
   * Currently used to model while(c) {} loop iteration counts
   */
  class UnknownOffsetVal : public OffsetVal {
    private:
    public:
      const llvm::Value* cause;
      UnknownOffsetVal(llvm::Value* v) : OffsetVal(OV_Unk), cause(v) {
        assert(v != nullptr);
      }
      bool isConst() const {return false;}
      const llvm::APInt& constVal() const;
      const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      void print(std::ostream& stream) const;

      static bool classof(const OffsetVal *ov) { return ov->getKind() == OV_Unk; }
  };

  /**
   * OffsetVal specialization for binary compound values
   */
  enum OffsetOperator {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Eq,
    Neq,
    SLT,
    SLE,
    SGT,
    SGE,
    ULT,
    ULE,
    UGT,
    UGE,
    end
  };

  class BinOpOffsetVal : public OffsetVal {
    private:
      const std::string getPrintOp() const;

    public:
      const OffsetValPtr lhs;
      const OffsetValPtr rhs;
      const OffsetOperator op;
      BinOpOffsetVal(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs) :
        OffsetVal(OV_BinOp), lhs(lhs), rhs(rhs), op(op) {
          assert(lhs != nullptr);
          assert(rhs != nullptr);
          assert(op != OffsetOperator::end);
        }
      bool isConst() const;
      bool isCompare() const;
      const llvm::APInt& constVal() const;
      const std::pair<llvm::APInt, llvm::APInt> constRange() const;
      void print(std::ostream& stream) const;

      static bool classof(const OffsetVal *ov) { return ov->getKind() == OV_BinOp; }
  };
}
#endif
