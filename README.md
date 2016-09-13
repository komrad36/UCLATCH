Fastest implementation of an UPRIGHT (no rotation) LATCH 512-bit binary feature descriptor as described in the 2015 paper by Levi and Hassner:

"LATCH: Learned Arrangements of Three Patch Codes" http://arxiv.org/abs/1501.03719

See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:

"The CUDA LATCH Binary Descriptor"

Note once again that this is an UPRIGHT CUDA LATCH, a.k.a. UCLATCH. A fast rotation- and scale-invariant version is in the works.

This implementation is insanely fast, matching or beating the much simpler ORB descriptor despite outputting twice as many bits AND being a superior descriptor.

A key insight responsible for much of the performance of this laboriously crafted CUDA kernel is due to Christopher Parker (https://github.com/csp256), to whom I am extremely grateful.

CUDA CC 3.0 or higher is required.

All functionality is contained in the files UCLATCH.h and UCLATCH.cu. 'main.cpp' is simply a sample test harness with example usage and performance testing.