CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**
* Haorong Yang
* [LinkedIn](https://www.linkedin.com/in/haorong-henry-yang/)
* Tested on: Windows 10 Home, i7-10750H @ 2.60GHz 16GB, GTX 2070 Super Max-Q (Personal)

The goal of this project was to implement a stream compaction algorithm on the GPU in CUDA from scratch. 
The algorithm will remove `0`s from an array of `int`s utilizing a scan function, which performs parallel reduction on the array to obtain an exclusive prefix sum.

Although the goal is to obtain an efficient parallel solution, for comparison, a few variations of the algorithm were also implemented.
A list of algorithms that will be compared to each other:
* CPU scan function
* CPU stream compaction without scan
* CPU sream compaction with scan
* GPU naive scan
* GPU work-efficient scan & compaction
* thrust library's implementation

The test results for array size of 2^8 is:

```
****************
** SCAN TESTS **
****************
    [  17  20  19  34  19   3   6   3  27   2  14   5  21 ...  36   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ... 6163 6199 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ... 6092 6101 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.029056ms    (CUDA Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ... 6163 6199 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.026752ms    (CUDA Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.012768ms    (CUDA Measured)
    [ 6199 6216 6236 6255 6289 6308 6311 6317 6320 6347 6349 6363 6368 ... 3050 3086 ]
    a[0] = 0, b[0] = 6199
    FAIL VALUE
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.012512ms    (CUDA Measured)
    [ 6138 6155 6175 6194 6228 6247 6250 6256 6259 6286 6288 6302 6307 ... 2979 2988 ]
    a[0] = 0, b[0] = 6138
    FAIL VALUE
==== thrust scan, power-of-two ====
   elapsed time: 0.055264ms    (CUDA Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ... 6163 6199 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.054368ms    (CUDA Measured)
    [   0  17  37  56  90 109 112 118 121 148 150 164 169 ... 6092 6101 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   1   2   1   1   0   1   1   0   2   1   3 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   3   1   2   1   1   1   1   2   1   3   1   2   2 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   3   1   2   1   1   1   1   2   1   3   1   2   2 ...   1   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.004ms    (std::chrono Measured)
    [   3   1   2   1   1   1   1   2   1   3   1   2   2 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.020992ms    (CUDA Measured)
    [   3   1   2   1   1   1   1   2   1   3   1   2   2 ...   0   0 ]
expected count is 185, count is 185
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.021888ms    (CUDA Measured)
    [   3   1   2   1   1   1   1   2   1   3   1   2   2 ...   1   3 ]
expected count is 185, count is 183
    passed
```