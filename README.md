# GPUCheck ![Build Monitor](https://travis-ci.org/taylorlloyd/GPUCheck.svg?branch=master)
Static Analyzer for GPU Performance Problems

## GPU Performance Problems

GPUCheck identifies 2 common sources of slowdowns in GPU programs: Noncoalescable memory accesses and divergent branches.

### Noncoalescable Memory Accesses

Threads within a GPU warp can combine memory accesses when addresses are within a single cache line, reducing pressure on the memory system and increasing instruction throughput. When this isn't possible, large delays can occur. GPUCheck symbolically inspects memory access addresses, and determines the number of accessed cache lines, warning when an access would be noncoalescable.

### Divergent Branches

GPUs execute all threads within a warp in lockstep, and when threads evaluate a conditional branch differently, the whole warp must execute both the taken and not-taken branches. GPUCheck warns when this is possible by inspecting branch conditions.

## Building

GPUCheck is built with CMake, and requires LLVM 5.0 to be present. Once
LLVM is installed, GPUCheck can be built with the following commands:

    mkdir build
    cd build
    cmake ..
    make

## Usage

GPUCheck is designed as a loadable LLVM pass module. Given a GPU executable in LLVM IR, GPUCheck can be run as follows:

    opt -load gpuchk/libGpuAnalysis.so -coalesce -bdiverge gpucode.bc
