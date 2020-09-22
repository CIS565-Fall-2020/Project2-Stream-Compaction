CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zijing Peng
  - [LinkedIn](https://www.linkedin.com/in/zijing-peng/)
  - [personal website](https://zijingpeng.github.io/)
* Tested on: Windows 22, i7-8750H@ 2.22GHz 16GB, NVIDIA GeForce GTX 1060

### Summary

In this project, I've implemented GPU stream compaction in CUDA, from scratch. The goal of stream compaction is to remove `0`s from an array of `int`s. The stream compaction includes three steps: map, scan and scatter. I've also implemented several exclusive scan algorithms, which produces a prefix sum array of a given array.

All the implementations including:

- Scan algorithms 
  - CPU Scan & Stream Compaction
  - Naive GPU Scan
  - Work-Efficient GPU Scan
  - Thrust Scan
- Stream Compaction algorithms 
  - CPU Stream Compaction
  - Work-Efficient GPU Stream Compaction

### Scan Performance Analysis

![](/img/scan.png)



The four scan algorithm implementations experience huge performance loss as the size of data increase, especially when the size is over 1M. The CPU scan has good performance when the dataset is small. It runs faster than 3 other scan when the size is smaller than 4K, but after that its performance lose rapidly. When the size increases to 16 M,it is much worse than all other GPU implementations. That is because GPU is designed for thousands of computation in parallel. CPU has limited threads, and there are some optimizations in the OS, so it could run pretty fast when the dataset is small. But when the dataset is super large, it will experience huge performance loss. 

Compared with naive scan, work-efficient scan is not so efficient, it even a little bit worse. As Part 5 discussed, more optimizations could be done to improve the performance of work-efficient scan.

Among the four implementations, thrust scan is undoubtedly the best. When the dataset is small, the advantage of thrust scan is not so obvious compared to others. However, it the only one that still has good performance when the size of dataset is 16M. I take a look at the Nsight timeline for its execution. I find there are several `cudaDeviceSynchronize` function calls, which means they use shared memory. Moreover, I find that in thrust implementation the kernel is only called once (while the up/down sweep of work-efficient is called 24 times with the same data size). The thrust scan use 40 registers per thread while my implementation only use 16 registers per thread. Thus, I guess it take advantage of shared memory and registers.

### Stream Compaction Performance Analysis

![](/img/compact.png)

The two implementations both experience huge performance loss as the size of data increase, especially when the size. As we have discussed above, the CPU implementation has good performance when the dataset is small and experience huge performance when dataset greatly increase. While the GPU compaction shows great performance when the size is over 1M.

### Output 

An out put when `SIZE = 256` and `blockSize = 512` .

```
****************
** SCAN TESTS **
****************
    [  39  42   3  23  38  47   7  10  32  49  44  21  25 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   0  39  81  84 107 145 192 199 209 241 290 334 355 ... 6130 6139 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  39  81  84 107 145 192 199 209 241 290 334 355 ... 6074 6110 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.018816ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.018272ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.081056ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.044896ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.054528ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.054112ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   3   1   0   3   3   0   2   3   2   3   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
    [   1   2   3   1   3   3   2   3   2   3   3   2   2 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   1   2   3   1   3   3   2   3   2   3   3   2   2 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.004ms    (std::chrono Measured)
    [   1   2   3   1   3   3   2   3   2   3   3   2   2 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.092768ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.125632ms    (CUDA Measured)
    passed

```









