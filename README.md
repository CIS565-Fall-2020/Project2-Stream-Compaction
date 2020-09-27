CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Weiyu Du
  * [LinkedIn](https://www.linkedin.com/in/weiyu-du/)
* Tested on: CETS virtual lab MOR100B-05 Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz

### Plots
1) Plot of time elapsed in (ms) versus array size when n is a power of 2 (x axis: 2^8, 2^12, 2^16, 2^20)
<img src="https://github.com/WeiyuDu/Project2-Stream-Compaction/blob/master/img/hw2_pow2.png"/>
2) Plot of time elapsed in (ms) versus array size when n is not a power of 2 (x axis: 2^8, 2^12, 2^16, 2^20)
<img src="https://github.com/WeiyuDu/Project2-Stream-Compaction/blob/master/img/hw2_nonpow2.png"/>

### Analysis
When the array size is small, we observe that cpu method is better than gpu ones and naive scan is best of the gpu ones. Possible explanations: 1) When array size is small, computation time difference is very small and accessing memory contributes to the largest portion of time. That's why gpu methods are worse than cpu. 2) Work efficient has up-sweep and down-sweep stages. Even though it has the same time complexity as naive method, constants matter with small n.

However, when array size increases, we observe that cpu performance quickly deteriorates and becomes than work efficient and thrust implementation. Among all the gpu methods, thrust is the fastest, work-efficient scan comes the second and naive scan is the slowest. This is as expected: 1) cpu method has run time complexity of O(n) while gpu methods have O(logn). Therefore, gpu performance is less susceptible to increase in array size. 2) Work efficient scan requires only one array while naive implementation has to access memory of two arrays. Global memory I/O is the bottleneck here, causing naive method (with heavy memory access) to be even worse than cpu. 3) Thrust utilizes shared memory while naive and work-efficient both uses global memory -- accessing shared memory is faster than accessing global memory. 

### Output
Array size is 2^20.
````

****************
** SCAN TESTS **
****************
    [  19  36  40  30  35  35  17   8  28  32  41  40  15 ...  44   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.7577ms    (std::chrono Measured)
    [   0  19  55  95 125 160 195 212 220 248 280 321 361 ... 25698986 25699030 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.9503ms    (std::chrono Measured)
    [   0  19  55  95 125 160 195 212 220 248 280 321 361 ... 25698890 25698926 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 2.7335ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 2.73654ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.32346ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.30934ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.405888ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.328032ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   3   0   2   1   1   2   1   0   3   1   2   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 4.1676ms    (std::chrono Measured)
    [   1   3   2   1   1   2   1   3   1   2   3   3   3 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.6659ms    (std::chrono Measured)
    [   1   3   2   1   1   2   1   3   1   2   3   3   3 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 10.0887ms    (std::chrono Measured)
    [   1   3   2   1   1   2   1   3   1   2   3   3   3 ...   1   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2.32755ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2.18624ms    (CUDA Measured)
    passed
````
