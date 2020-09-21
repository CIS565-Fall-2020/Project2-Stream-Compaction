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

Add [csvfile.hpp]() to get the performance in CSV form. 

### Intro

In this project, basically we implement parallel scan algorithm based on CUDA required by [instruction](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/INSTRUCTION.md). 



#### Part 1~4:



#### Part 5: why GPU version so slow

The reason why the GPU is slower than CPU version:

1. **Spatial coherence:** The cpu version reads the memory in a continuous way while the current version fetches memory uncontinuously, which leads to a low memory bandwidth.  
2. **The input size matters:** When the size of input array is trivial (for example 2^4), **cpu** version is faster than **gpu's**. When the size goes up, the situation goes reversed and **gpu** version is much faster than **cpu's** since naturally **gpu** is better in dealing with a large amounts of number.

