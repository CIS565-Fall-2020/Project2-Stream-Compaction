CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Qiaosen Chen
  * [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/), etc.
* Tested on: Windows 10, i5-9400 @ 2.90GHz 16GB, GeForce RTX 2060 6GB (personal computer).

## Implemented Features

This project includes several scan and stream compaction algorithms, and most of them are implemented with parallelism in CUDA. With the serial version of algorithms run on CPU as comparison, we can have a better view on the performance of parallel algorithm run on GPU. 

- CPU Scan & Stream Compaction

- Naive GPU Scan Algorithm

- Work-Efficient GPU Scan & Stream Compaction

- Thrust Scan

- Radix Sort (Extra Credit)

  Please see the details in the last part of the report.

## Performance Analysis

### Performances under Different Block Size 

![Different Block Size](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/img/different_blocksize_perf.png)

In general, when ```blockSize = 128```, the parallel version of scan algorithms and compact algorithms could achieve a relative optimized performance.

### Scan Algorithms Performances

![Scan Algorithms Performances](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/img/different_arraysize_scan_perf.png)

This performance analysis tested when ```blockSize = 256```, and all algorithms taken into accounts are given the input with an power-of-two array size.

When the input array size is small (```SIZE < 2^15```), the difference of performances is small and not obvious for all the scan algorithms, but the serial version of algorithm run on CPU performs better than those parallel version of algorithms run on GPU. 

When the input array size is large enough (```SIZE > 2^17```), the difference of performances becomes larger and larger, and apparently at this time, ```Thrust::Scan``` performs best among all algorithms. As expected, the ```CPU::Scan``` algorithm performs much worse than ```Thrust::Scan```, it is even worse than ```Naive::Scan```algorithm. However, it is quite weird that, the naive scan algorithm always runs faster than the work-efficient scan algorithm, because the so-called "efficient" work-efficient scan algorithm can still get optimized.  

### Compact Algorithm Performances

![Compact Algorithm Performances](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/img/different_arraysize_compact_perf.png)

When the input array size is large enough (```SIZE > 2^17```), the difference of performances become more and more obvious, and both of the compact algorithms perform much worse as the input size increases, as expected. The serial version of compact algorithm run on CPU performs better than the parallel version run in GPU when the input size is small, however, when the input size is quite huge, such as ```SIZE = 2^20```, there is no doubt that the work-efficient compact algorithm run in parallel on GPU perform much better than the CPU version.

### Output

This output tests were based on an array ```SIZE = 1024``` and ```blockSize = 128```:

```bash
****************
** SCAN TESTS **
****************
    [  46  46  37   6  29   0  28  22  25  23   3  11  29 ...  20   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   0  46  92 129 135 164 164 192 214 239 262 265 276 ... 24775 24795 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   0  46  92 129 135 164 164 192 214 239 262 265 276 ... 24725 24751 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.02192ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.021952ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.046304ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.045856ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.038912ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.038176ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   1   2   3   2   0   0   1   1   3   1   1 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0026ms    (std::chrono Measured)
    [   2   1   2   3   2   1   1   3   1   1   1   2   1 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0024ms    (std::chrono Measured)
    [   2   1   2   3   2   1   1   3   1   1   1   2   1 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0054ms    (std::chrono Measured)
    [   2   1   2   3   2   1   1   3   1   1   1   2   1 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.062976ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.052992ms    (CUDA Measured)
    passed
```

## Extra Credit

- **Radix sort**

  I defined the several function used by Radix Sort in [radix.h](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/stream_compaction/radix.h) and implemented it in [radix.cu](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/stream_compaction/radix.cu) under the directory [/stream_compaction](https://github.com/giaosame/Project2-Stream-Compaction/tree/master/stream_compaction). In [main.cpp](https://github.com/giaosame/Project2-Stream-Compaction/blob/master/src/main.cpp), I called this function ```StreamCompaction::Radix::sort``` in the last of the ```main``` function.

  ```c++
  zeroArray(SIZE, c);
  printDesc("radix sort, power-of-two");
  StreamCompaction::CPU::sort(SIZE, b, a);
  StreamCompaction::Radix::sort(SIZE, c, a);
  printCmpResult(SIZE, b, c);
  
  zeroArray(SIZE, c);
  printDesc("radix sort, non-power-of-two");
  StreamCompaction::CPU::sort(NPOT, b, a);
  StreamCompaction::Radix::sort(NPOT, c, a);
  printCmpResult(NPOT, b, c);
  ```

  Examples of output of Radix Sort:

  ```bash
  **********************
  ** RADIX SORT TESTS **
  **********************
      [  10  31  19  93  79  96  60  46  46  85  44  56  52  53  85  39 ]
  ==== radix sort, power-of-two ====
      [  10  19  31  39  44  46  46  52  53  56  60  79  85  85  93  96 ]
      passed
  ==== radix sort, non-power-of-two ====
      [  10  19  31  44  46  46  52  56  60  79  85  93  96 ]
      passed
  ```

  

  

