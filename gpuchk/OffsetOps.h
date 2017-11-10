#include "OffsetVal.h"
#include "ThreadDepAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constant.h"
#include <unordered_map>

#ifndef OFFSET_OP_H
#define OFFSET_OP_H

namespace gpucheck {
  OffsetValPtr sumOfProducts(OffsetValPtr ov);
  OffsetValPtr sumOfProductsPass(OffsetValPtr ov);
  OffsetValPtr negateCondition(OffsetValPtr& cond);
  OffsetValPtr simplifyOffsetVal(OffsetValPtr ov);
  OffsetValPtr cancelDiffs(OffsetValPtr ov, ThreadDependence& td);
  OffsetValPtr simplifyDifferenceOfProducts(OffsetValPtr addt, OffsetValPtr subt, ThreadDependence& td);
  OffsetValPtr simplifyConstantSubExpressions(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs);

  OffsetValPtr replaceComponents(const OffsetValPtr& orig, std::unordered_map<OffsetValPtr, OffsetValPtr>& rep);
  OffsetValPtr simplifyConstantVal(OffsetValPtr lhs, OffsetOperator op, OffsetValPtr rhs);
  bool matchingOffsets(OffsetValPtr lhs, OffsetValPtr rhs);
}
#endif
