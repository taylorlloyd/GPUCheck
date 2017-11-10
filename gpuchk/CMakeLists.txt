add_library(GpuAnalysis MODULE ThreadDepAnalysis.cpp
                               BugEmitter.cpp
                               Utilities.cpp
                               OffsetVal.cpp
                               OffsetPropagation.cpp
                               OffsetOps.cpp
                               BranchDivergeAnalysis.cpp
                               MemCoalesceAnalysis.cpp
                               AddrSpaceAnalysis.cpp
)
