#include "llvm/IR/Instruction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "BugEmitter.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cxxabi.h>

#define MACHINE_READABLE

using namespace llvm;
using namespace std;

inline string demangle(string name) {
  int status = -1;
  unique_ptr<char, void(*)(void*)> res { abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free };
        return (status == 0) ? res.get() : string(name);
}

bool printline(string filename, int lineNumber) {
  ifstream infile(filename);
  string line;
  int linenum = 0;

  // Load to the appropriate line
  while(linenum < lineNumber && getline(infile, line)) {
    linenum++;
  }
  if(linenum != lineNumber)
    return false;

  errs()<< "    " << line << "\n";
  return true;
}

void gpucheck::emitWarning(string warning, Instruction* i, Severity sev) {
  DILocation *Loc = i->getDebugLoc();
#ifdef MACHINE_READABLE
  if(!Loc)
    return;
  errs() << Loc->getFilename() << ":" << Loc->getLine() << "\n";
#else
  string funcName = demangle(i->getParent()->getParent()->getName().str());
  string sevStr;
  switch(sev) {
    case SEV_UNKNOWN:
      sevStr = "(Unk) ";
      break;
    case SEV_MAX:
      sevStr = "(MAX) ";
      break;
    case SEV_MED:
      sevStr = "(Med) ";
      break;
    case SEV_MIN:
      sevStr = "(min) ";
      break;
  }
  if(!Loc) {
    errs() << sevStr << "Warning: " << warning << "\n";
    errs() << "in " << funcName << ":\n";
    i->dump();
    errs() << "\n";
  } else {
    errs() << sevStr << "Warning: " << warning << "\n";
    errs() << Loc->getFilename() << ":" << Loc->getLine() << " in " << funcName << ":\n";
    printline((Loc->getDirectory()+"/"+Loc->getFilename()).str(), Loc->getLine());
    errs() << "\n";

  }
#endif
}
