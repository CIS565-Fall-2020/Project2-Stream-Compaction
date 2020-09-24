CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**
* Haorong Yang
* [LinkedIn](https://www.linkedin.com/in/haorong-henry-yang/)
* Tested on: Windows 10 Home, i7-10750H @ 2.60GHz 16GB, GTX 2070 Super Max-Q (Personal)

The goal of this project was to implement a stream compaction algorithm on the GPU in CUDA from scratch. 
The algorithm will remove '0's from an array of 'int's utilizing a scan function, which performs parallel reduction on the array to obtain an exclusive prefix sum.

Although the goal is to obtain an efficient parallel solution, for comparison, a few variations of the algorithm were also implemented:
* CPU scan function
* CPU stream compaction without scan
* CPU sream compaction with scan
* GPU naive scan
* GPU work-efficient scan & compaction

Also the thrust library's implementation is also used for comparison.
