CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Questions and Plots

### Test Program Output

Array size = 1 << 8

```
****************
** SCAN TESTS **
****************
    [  22   1  25  15   7  27  27  23  12   1  49  11  46 ...  19   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  22  23  48  63  70  97 124 147 159 160 209 220 ... 6133 6152 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  22  23  48  63  70  97 124 147 159 160 209 220 ... 6088 6092 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.009216ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.008192ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.013312ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.012288ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.091968ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.053248ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   3   3   3   3   1   3   0   3   3   3   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   1   3   3   3   3   1   3   3   3   3   2   1   1 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0012ms    (std::chrono Measured)
    [   1   3   3   3   3   1   3   3   3   3   2   1   1 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0077ms    (std::chrono Measured)
    [   1   3   3   3   3   1   3   3   3   3   2   1   1 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.017408ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.017408ms    (CUDA Measured)
    passed
```
