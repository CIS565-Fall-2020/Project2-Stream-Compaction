CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

## Debug
1. Navie Scan

- I ping-pong buffers to keep arr1 as the input data and arr2 as the output data. But I found it inefficient. So I use a flag to denote which one is the input arr at each iteration.
- I need to determine which array is the final result. At first, I thought dev_arr2 will always be the result. But as I change the array size, dev_arr1 and dev_arr2 are both possible answers.

2. Efficient scan: My program works well with array size < 16 but crashed with array size >= 16. Error: Illigal memory
- Bug1: I forget to free the device buffers.
- Bug2(Main problem): Firstly, I try to avoid mod operations when implementing the kernUpSweep and kernDownSweep. Unluckily, it looks like my implementation has problems. So I still use mod operation to determine whether the current thread should do the calculations.

