CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Janine Liu
  * [LinkedIn](https://www.linkedin.com/in/liujanine/), [personal website](https://www.janineliu.com/).
* Tested on: Windows 10, i7-10750H CPU @ 2.60GHz 16GB, GeForce RTX 2070 8192 MB (personal computer)

### GPU Stream Compaction

This project involved implementing Scan and Compact algorithms that will be used for in later projects, comparing their performance on both the GPU and CPU with different array sizes. In detail, this project includes a CPU version of Scan and Compact (both serialized), a naive version of Scan, a work-efficient version of Scan, and a work-efficient version of Compact that used the work-efficient Scan's code. The Thrust version of Scan is also compared with the rest of these algorithms as an additional reference. 


## Performance Analysis Methods

The CPU and GPU algorithms were timed during their execution, and their times are written to a formatted output that is printed to the terminal. An example of that output is as follows:

```
****************
** SCAN TESTS **
****************
    [  18  44  40  43   2  39  23  12   8   5  11  16  31 ...   5   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.008ms    (std::chrono Measured)
    [   0  18  62 102 145 147 186 209 221 229 234 245 261 ... 100690 100695 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0069ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.04208ms    (CUDA Measured)
    [   0  18  62 102 145 147 186 209 221 229 234 245 261 ... 100690 100695 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.038656ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.093504ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.083968ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.04832ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.04752ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   2   3   2   3   1   2   0   1   3   2   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0082ms    (std::chrono Measured)
    [   2   2   3   2   3   1   2   1   3   2   1   3   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0082ms    (std::chrono Measured)
    [   2   2   3   2   3   1   2   1   3   2   1   3   3 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0287ms    (std::chrono Measured)
    [   2   2   3   2   3   1   2   1   3   2   1   3   3 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.124928ms    (CUDA Measured)
    [   2   2   3   2   3   1   2   1   3   2   1   3   3 ...   2   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.122176ms    (CUDA Measured)
    passed
```

To collect data, I run the program five times in succession and record the timed values for the power of 2 arrays in each implementation. Each run, the program generates a new random array of values that all of the algorithms operate on. I varied the size of the arrays by powers of 2 

## Runtime Analysis

fluctuated too much to tell. difference between 32 and 64 

Thurst Scan and work-efficient scan fluctuates on non power of two arrays. for insance, 1024 array size block size 128, Thrust 0.231424	 <- non power of two case