CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

A project to implement and analyze the performance of Scan (prefix-sum) on the GPU. In this repo, I implement scan on the CPU as well as on the GPU in two ways, a naive approach and a work efficient approach. I also apply the scan algorithm to perform stream compaction using a scan-scatter approach. Then, I compare the performance of my implementations of scan with the reference thrust implementation provided by Nvidia. Lastly, I compare my different approaches for stream compaction: CPU naive approach, CPU using scan-scatter, and GPU using my work-efficient scan-scatter approach. 

NAME: CHETAN PARTIBAN 

GPU: GTX 970m (Compute Capability 5.2) 

Tested on Windows 10, i7-6700HQ @ 2.60 GHz 16Gb, GTX 970m 6Gb (Personal Laptop) 

## Performance Analysis
**Block Size Analysis**


**Example Program Output:**

```
****************** SCAN TESTS ******************                                                                                                            
[  20  28  13  32  32   2   1  19  40  25  34  49  23 ...  43   0 ]                                                 
==== cpu scan, power-of-two ====                           elapsed time: 6.03387ms    (std::chrono Measured)                                                                    
==== cpu scan, non-power-of-two ====                       elapsed time: 6.10951ms    (std::chrono Measured)              passed                                                                                                             
==== naive scan, power-of-two ====                         elapsed time: 2.40527ms    (CUDA Measured)                     passed                                                                                                              
==== naive scan, non-power-of-two ====                     elapsed time: 2.39343ms    (CUDA Measured)                     passed                                                                                                              
==== work-efficient scan, power-of-two ====                elapsed time: 0.971063ms    (CUDA Measured)                    passed                                                                                                             
==== work-efficient scan, non-power-of-two ====            elapsed time: 0.971972ms    (CUDA Measured)                    passed                                                                                                              
==== thrust scan, power-of-two ====                        elapsed time: 0.523031ms    (CUDA Measured)                    passed                                                                                                              
==== thrust scan, non-power-of-two ====                    elapsed time: 0.434082ms    (CUDA Measured)                    passed                                                                                                                                                                                                                                      
******************************* STREAM COMPACTION TESTS *******************************                                                                                             
[   1   2   3   1   2   3   0   2   3   1   2   3   0 ...   1   0 ]                                                 
==== cpu compact without scan, power-of-two ====           elapsed time: 3.71089ms    (std::chrono Measured)              passed                                                                                                              
==== cpu compact without scan, non-power-of-two ====       elapsed time: 3.71827ms    (std::chrono Measured)              passed                                                                                                              
==== cpu compact with scan ====                            elapsed time: 16.5354ms    (std::chrono Measured)              passed                                                                                                              
==== work-efficient compact, power-of-two ====             elapsed time: 1.25834ms    (CUDA Measured)                     passed                                                                                                              
==== work-efficient compact, non-power-of-two ====         elapsed time: 1.24629ms    (CUDA Measured)                     passed      
```
