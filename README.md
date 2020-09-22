CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ling Xie
  * [LinkedIn](https://www.linkedin.com/in/ling-xie-94b939182/), 
  * [personal website](https://jack12xl.netlify.app).
* Tested on: 
  * Windows 10, Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 2.20GHz ( two processors) 
  * 64.0 GB memory
  * NVIDIA TITAN XP GP102

Thanks to [FLARE LAB](http://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/flare/index.html) for this ferocious monster.

##### Cmake change

Add 

1. [csvfile.hpp]() to automatically record the performance in CSV form. 
2. [radixSort.h](), [radixSort.]() for [Part 6]()

#### Intro

In this project, basically we implement **scan** algorithm(in specific prefix sum) based on CUDA parrallelism, as required by [instruction](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/INSTRUCTION.md). The detailed algorithm content can be viewed at [nvidia gpu gem](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

Here we managed to implement **all** the compulsory and extra point sections.



#### Overview

##### Optimized block size

After implementing all the functions, first we try to find the optimized block size.

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/2_x_baseline.png)

From the image, we may choose the optimized block size as 256

##### Implementation Comparisons

Here we compare each scan, compact implementations under different array size. The results below are ran under block size = 256. 

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/2_x_baseline.png)

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/2_x_baseline.png)

##### notes:

- the **non-opt** refers to the non-optimization scan function before Part 5.
- The **idx** refers to the optimized version in Part 5.
- The shared memory refers to the optimized version in Part 7

##### Output of test program

Here we add test for radix sort, shared memory based scan and compact.

```

****************
** SCAN TESTS **
****************
    [   1   0   0   1   1   1   0   1   1   1   0   0   1 ...   0   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0589ms    (std::chrono Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32801 32801 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.056ms    (std::chrono Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32799 32800 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.042656ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32801 32801 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.041024ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.104704ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32801 32801 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.108032ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32799 32800 ]
    passed
==== work-efficient scan with shared memory, power-of-two ====
   elapsed time: 0.0256ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32801 32801 ]
    passed
==== work-efficient scan with shared memory, non-power-of-two ====
   elapsed time: 0.025024ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32799 32800 ]
    passed
==== work-efficient scan with index scale, power-of-two ====
   elapsed time: 0.083584ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32801 32801 ]
    passed
==== work-efficient scan with index scale, non-power-of-two ====
   elapsed time: 0.077536ms    (CUDA Measured)
    [   0   1   1   1   2   3   4   4   5   6   7   7   7 ... 32799 32800 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.094944ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.091936ms    (CUDA Measured)
    passed

*****************************
** STREAM SORT TESTS **
*****************************
    [  31  24  18   1  17  25   4  15  25   3  30  22   7 ...   5   0 ]
The array to be sorted is :
    [  31  24  18   1  17  25   4  15  25   3  30  22   7 ...   5   0 ]
==== Std sort ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   0   1   1   1   3   4   5   5   5   5   7   7   9 ...  30  31 ]
==== Radix sort ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   0   1   1   1   3   4   5   5   5   5   7   7   9 ...  30  31 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   2   1   1   1   0   3   1   3   2   2   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.2151ms    (std::chrono Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.4586ms    (std::chrono Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.5532ms    (std::chrono Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.443296ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   2   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.403328ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   1   2 ]
    passed
==== work-efficient compact with idx mapping, power-of-two ====
   elapsed time: 0.362304ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   2   1 ]
    passed
==== work-efficient compact with idx mapping, non-power-of-two ====
   elapsed time: 0.493792ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   1   2 ]
    passed
==== work-efficient compact with shared memory, power-of-two ====
   elapsed time: 0.394784ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   2   1 ]
    passed
==== work-efficient compact with shared memory, non-power-of-two ====
   elapsed time: 0.463968ms    (CUDA Measured)
    [   3   2   1   1   1   3   1   3   2   2   3   3   2 ...   1   2 ]
    passed
```



#### Part 1~3:

The performance is showed in previous image.

#### Part 4:

Here shows the thrust summary and timeline:

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/2_x_baseline.png)

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/2_x_baseline.png)

#### Part 5: why GPU version so slow [Extra point]

The reason why the GPU is slower than CPU version:

1. **Spatial coherence:** The cpu version reads the memory in a continuous way while the current version fetches memory uncontinuously, which leads to a low memory bandwidth.  
2. **The input size matters:** When the size of input array is trivial (for example 2^4), **cpu** version is faster than **gpu's**. When the size goes up, the situation goes reversed and **gpu** version is much faster than **cpu's** since naturally **gpu** is better in dealing with a large amounts of number.
3. **Occupancy low**: Not all the threads are doing its job in non-optimization version.

##### Simple solution

We can increase the **Occupancy** by mapping the each thread index to the active index to force them to work. Also, we can dynamically adjust the grid dimension to stop calling useless grid.

##### Tips:

The mapping step may cause integer overflow. So we use **size_t** for thread index.

#### Part 6 Radix Sort [Extra point]

For simplicity and less memory copy between gpu and cpu, we mainly implement the algorithm in cpu side, except for the scan function, which we call the shared memory version from part 7.

We compare the results with built-in std::sort. Here we show the correctness of the radix sort.

```
*****************************
** STREAM SORT TESTS **
*****************************
    [  31  24  18   1  17  25   4  15  25   3  30  22   7 ...   5   0 ]
The array to be sorted is :
    [  31  24  18   1  17  25   4  15  25   3  30  22   7 ...   5   0 ]
==== Std sort ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   0   1   1   1   3   4   5   5   5   5   7   7   9 ...  30  31 ]
==== Radix sort ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   0   1   1   1   3   4   5   5   5   5   7   7   9 ...  30  31 ]
    passed

```



#### Part 7 Scan with shared memory [Extra point]

As is showed in previous figure, adding shared memory can boost the performance in a large degree since it provides higher memory bandwidth than global memory. 

##### Implementation explain

First the implementation in gpu gem is somehow not that robust to input blocksize because it tries to read all block memory into a single shared memory bank. If the blocksize keep increasing, it would soon drain the shared memory limit(48 kb) .

So instead, we tear the up-sweep and down-sweep process in several part( based on the block size) with different sweep depth. In each part we respectively assign the shared memory based on the largest array this part would hop into.

##### Detail:

In our design, we set the shared memory size twice as big as the block size. The reason for this is to utilize the index mapping from [part 5]().

Sadly we do not consider the bank conflict effect.

##### Tips

The shared memory version is prone to cause integer overflow so we decrease the element range in input array.



#### Questions:

- ##### Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.

  - As is discussed in [here](), we adopt the 256 block size for both naive and efficient version.

- ##### Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).

  - The picture is showed [here](). 

- ##### Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

  - Personally I believe the bottlenecks lie mainly in memory I/O. Because for each implementation the computation is pretty straight(with complexity **O(n)** and **O(n * log(n)**). When the shared memory is introduced, the performance goes up drastically.  

- ##### Paste the output of the test program into a triple-backtick block in your README.

  - Pasted [here]()

