CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shenyue Chen
  * [LinkedIn](https://www.linkedin.com/in/shenyue-chen-5b2728119/), [personal website](http://github.com/EvsChen)
* Tested on: Windows 10, Intel Xeon Platinum 8259CL @ 2.50GHz 16GB, Tesla T4 (AWS g4dn-xlarge)

### Features
* Implementation of cpu scan, cpu compact, naive scan, work-efficient scan and work-efficient compact
* Optimization of the work-efficient scan algorithm by launching only necessary number of threads in `upSweep` and `downSweep`
    * Map thread index to the actual index by `interval * idx + interval - 1`, where interval is `1 << iteration`
    * For example, for N = 8, 4 threads is launched in the first iteration, 2 launched in the secondm etc.

### Performance analysis
Block size: 128

Lengh of array: from (2^10 - 2^21)

**All the time measured in this section is the average of 100 tests, to avoid caching of functions**

As N increases, the time for cpu algorithms increases in a exponential manner while the GPU algorithms increases much slower.

<p align="center">
<image src="doc/scan_time.png">
</p>

<p align="center">
<image src="doc/compact.png">
</p>

For the GPU algorithms only, the naive algorithm performs the best when N is small. But thrust scan turns out to be the best when N becomes larger.
<p align="center">
<image src="doc/gpu_scan.png">
</p>

In my experiments, there are no obvious difference for the NPOT version of the work efficient scan algorithm.
<p align="center">
<image src="doc/scan_time_npot.png">
</p>

Similar things happen for the thrust scan.
<p align="center">
<image src="doc/thrust_scan_time_npot.png">
</p>




### Sample output
I tested each of the algorithm for 100 times and include some additional information.
```
****************
** SCAN TESTS **
****************
    [  41  23  25   1   5  46  28  37  30  42  42  25  35 ...  38   0 ]
==== cpu scan, power-of-two ====
    Time record is [11.006, 11.431, 11.774, 13.613, 14.736, 11.656, 11.874, 11.659, 11.823, 12.829, 11.737, 11.989, 11.872, ... 25.619]
   elapsed time: 13.311ms    (std::chrono Measured)
    [   0  41  64  89  90  95 141 169 206 236 278 320 345 ... 51331714 51331752 ]
==== cpu scan, non-power-of-two ====
    Time record is [6.4447, 6.2375, 20.608, 8.2194, 4.5156, 5.2029, 4.3476, 3.7569, 3.7672, 3.9463, 3.8041, 6.7647, 8.9613, ... 3.7673]
   elapsed time: 4.7256ms    (std::chrono Measured)
    [   0  41  64  89  90  95 141 169 206 236 278 320 345 ... 51331652 51331692 ]
    passed
==== naive scan, power-of-two ====
    Time record is [1.6997, 1.6947, 1.6957, 1.6964, 1.6937, 1.6972, 1.6957, 1.6955, 1.6977, 1.6956, 1.6957, 1.697, 1.6957, ... 1.5053]
   elapsed time: 1.5804ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
    Time record is [1.505, 1.5154, 1.5114, 1.5073, 1.5134, 1.5131, 1.5095, 1.5173, 1.5183, 1.5173, 1.5181, 1.5193, 1.5177, ... 1.5286]
   elapsed time: 1.5617ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
    Time record is [1.0465, 0.9728, 0.92182, 0.9257, 0.93184, 2.3247, 0.92374, 0.9345, 0.92896, 0.92176, 0.92269, 0.92266, 0.93555, ... 0.9264]
   elapsed time: 0.97037ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
    Time record is [0.93405, 0.92058, 0.92365, 0.91706, 0.93229, 0.92541, 0.9175, 0.93056, 0.9216, 0.9176, 0.91802, 0.93424, 0.91955, ... 0.91955]
   elapsed time: 0.92851ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
    Time record is [0.27674, 0.29424, 0.37693, 0.28387, 0.26618, 0.26726, 0.29562, 0.27222, 0.27443, 0.29901, 0.31325, 0.30925, 0.27082, ... 0.26301]
   elapsed time: 0.30972ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
    Time record is [0.2639, 0.36099, 0.28035, 0.26765, 0.39936, 0.30883, 0.29104, 0.27306, 0.26934, 0.2631, 0.29914, 0.26224, 0.2863, ... 0.29773]
   elapsed time: 0.31379ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   1   3   0   1   0   2   3   3   1   1   1 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
    Time record is [7.5451, 5.9954, 5.9785, 5.7061, 6.2471, 6.2241, 5.9236, 5.7947, 6.4508, 5.8406, 5.8629, 5.7438, 5.7671, ... 5.7358]
   elapsed time: 6.2666ms    (std::chrono Measured)
    [   1   1   3   1   2   3   3   1   1   1   3   1   2 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    Time record is [7.3508, 8.2913, 5.9938, 5.9339, 5.8238, 5.7023, 5.9023, 5.8208, 6.926, 6.5035, 5.7348, 6.8928, 7.7841, ... 5.8204]
   elapsed time: 6.2747ms    (std::chrono Measured)
    [   1   1   3   1   2   3   3   1   1   1   3   1   2 ...   2   3 ]
    passed
==== cpu compact with scan ====
    Time record is [28.751, 24.819, 23.735, 23.725, 23.829, 23.911, 23.968, 26.001, 24.981, 24.231, 23.606, 24.335, 23.51, ... 26.916]
   elapsed time: 24.973ms    (std::chrono Measured)
    [   1   1   3   1   2   3   3   1   1   1   3   1   2 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
    Time record is [1.9505, 1.6153, 1.6267, 1.6086, 1.6013, 1.6535, 1.6727, 1.7992, 1.7735, 2.1627, 1.7832, 1.7375, 1.7575, ... 1.682]
   elapsed time: 1.8556ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
    Time record is [1.6855, 1.9108, 1.7044, 1.6872, 1.7303, 1.9761, 1.7651, 1.9041, 1.6835, 1.6977, 1.7791, 1.6812, 1.7178, ... 1.6495]
   elapsed time: 1.7805ms    (CUDA Measured)
    passed
Result for n = 2097152 is :
    Time record is [13.311, 4.7256, 1.5804, 1.5617, 0.97037, 0.92851, 0.30972, 0.31379, 6.2666, 6.2747, 24.973, 1.8556, 1.7805, ... 0]
```



