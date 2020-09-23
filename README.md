CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* SPENCER WEBSTER-BASS
  * [LinkedIn](https://www.linkedin.com/in/spencer-webster-bass/)
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro P1000 222MB (MOR100B-19)

### DESCRIPTION

This project is an implementation of the stream compaction parallel algorithm on the GPU using CUDA and C++.

Features:
* Serial implementation of scan and stream compaction algorithms on the CPU
* Naive, parallel implementation of scan and stream compaction algorithms on the GPU
* Atepted work-efficient, parallel implementation of scan and stream compaction algorithms on the GPU
* Comparison between my implementations' efficiency and thrust's implementation of exclusive scan algorithm

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

****************
** SCAN TESTS **
****************
    [  49   8   2  27  20  44  21  27  49   3  20   3  16   3  31   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0ms    (std::chrono Measured)
    [   0  49  57  59  86 106 150 171 198 247 250 270 273 289 292 323 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0ms    (std::chrono Measured)
    [   0  49  57  59  86 106 150 171 198 247 250 270 273 ]
    passed
==== naive scan, power-of-two ====
    [  49   8   2  27  20  44  21  27  49   3  20   3  16   3  31   0 ]
    [ 49 8 2 27 20 44 21 27 49 3 20 3 16 3 31 0 ]
    [ 0 49 57 10 29 47 64 65 48 76 52 23 23 19 19 34 ]
    [ 0 49 57 59 86 57 93 112 112 141 100 99 75 42 42 53 ]
    [ 0 49 57 59 86 106 150 171 198 198 193 211 187 183 142 152 ]
    [ 0 49 57 59 86 106 150 171 198 247 250 270 273 289 292 323 ]
   elapsed time: 7.58922ms    (CUDA Measured)
    [   0  49  57  59  86 106 150 171 198 247 250 270 273 289 292 323 ]
    passed
==== 1s array for finding bugs ====
    [  49   8   2  27  20  44  21  27  49   3  20   3  16   3  31   0 ]
    [ 49 8 2 27 20 44 21 27 49 3 20 3 16 3 31 0 ]
    [ 0 49 57 10 29 47 64 65 48 76 52 23 23 19 19 34 ]
    [ 0 49 57 59 86 57 93 112 112 141 100 99 75 42 42 53 ]
    [ 0 49 57 59 86 106 150 171 198 198 193 211 187 183 142 152 ]
    [ 0 49 57 59 86 106 150 171 198 247 250 270 273 289 292 323 ]
    [   0  49  57  59  86 106 150 171 198 247 250 270 273 289 292 323 ]
==== naive scan, non-power-of-two ====
    [  49   8   2  27  20  44  21  27  49   3  20   3  16   3  31   0 ]
    [ 49 8 2 27 20 44 21 27 49 3 20 3 16 ]
    [ 0 49 57 10 29 47 64 65 48 76 52 23 23 3 0 0 ]
    [ 0 49 57 59 86 57 93 112 112 141 100 99 75 26 23 3 ]
    [ 0 49 57 59 86 106 150 171 198 198 193 211 187 167 123 102 ]
    [ 0 49 57 59 86 106 150 171 198 247 250 270 273 273 273 273 ]
   elapsed time: 15.0825ms    (CUDA Measured)
    [   0  49  57  59  86 106 150 171 198 247 250 270 273   0   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0ms    (CUDA Measured)
    a[1] = 49, b[1] = 0
    FAIL VALUE
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0ms    (CUDA Measured)
    a[1] = 49, b[1] = 0
    FAIL VALUE
==== thrust scan, power-of-two ====
   elapsed time: 0.083008ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.069632ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   2   1   2   0   1   3   1   3   0   3   0   3   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0034ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.004ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0023ms    (std::chrono Measured)
    [ ]
    expected 12 elements, got -1
    FAIL COUNT
==== work-efficient compact, power-of-two ====
   elapsed time: 0ms    (CUDA Measured)
    expected 12 elements, got -1
    FAIL COUNT
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0ms    (CUDA Measured)
    expected 10 elements, got -1
    FAIL COUNT
Press any key to continue . . .
