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

### Intro

In this project, basically we implement parallel scan algorithm based on CUDA required by [instruction](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/INSTRUCTION.md). 



#### Part 5: why GPU version so slow

The reason why the 

1. memory bandwidth: the current version fetches memory uncontinuously, which leads to a low memory bandwidth.  

