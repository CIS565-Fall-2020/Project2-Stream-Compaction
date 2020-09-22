CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Name: Gizem Dal
  * [LinkedIn](https://www.linkedin.com/in/gizemdal), [personal website](https://www.gizemdal.com/)
* Tested on: Predator G3-571 Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz 2.81 GHz - Personal computer (borrowed my friend's computer for the semester)

**Project Description**

The main focus of this project is implementing GPU stream compaction and other parallel algorithms in CUDA which are widely used and important for accelerating path tracers and algorithmic thinking. I implemented a few different versions of the Scan (Prefix Sum) algorithm including CPU scan, naive GPU scan, work-efficient GPU scan and GPU Thrust library scan. Then, I used some of these scan implementations to implement stream compaction for CPU and GPU. All of these implementations are timed in order to show runtime comparisons between different approaches and do a comprehensive runtime analysis in the following section.

**Performance Analysis**

Both CPU and GPU timing functions are warapped up as a performance timer class in order to measure the time cost conveniently. We use *std::chrono* to provide CPU high-precision timing and CUDA event to measure the CUDA performance. In order to measure the performance of the implementation itself, I **have not** included the cost of initial and final memory operations on the GPU such as cudaMalloc() or cudaMemcpy().

```****************
** SCAN TESTS **
****************
    [  46  27   0  28  19  24  10  37   8  21   7  41  15 ...  30   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.001ms    (std::chrono Measured)
    [   0  46  73  73 101 120 144 154 191 199 220 227 268 ... 5999 6029 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  46  73  73 101 120 144 154 191 199 220 227 268 ... 5953 5975 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.019456ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.019456ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.104448ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.120832ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   0   0   1   0   0   3   2   1   1   3   3 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   3   1   3   2   1   1   3   3   2   3   3   2   1 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   3   1   3   2   1   1   3   3   2   3   3   2   1 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0033ms    (std::chrono Measured)
    [   3   1   3   2   1   1   3   3   2   3   3   2   1 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.041984ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.171008ms    (CUDA Measured)
    passed
```
