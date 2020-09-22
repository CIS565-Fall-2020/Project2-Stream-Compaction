CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

A project to implement and analyze the performance of Scan (prefix-sum) on the GPU. In this repo, I implement scan on the CPU as well as on the GPU in two ways, a naive approach and a work efficient approach. I also apply the scan algorithm to perform stream compaction using a scan-scatter approach. Then, I compare the performance of my implementations of scan with the reference thrust implementation provided by Nvidia. Lastly, I compare my different approaches for stream compaction: CPU naive approach, CPU using scan-scatter, and GPU using my work-efficient scan-scatter approach. 

NAME: CHETAN PARTIBAN 

GPU: GTX 970m (Compute Capability 5.2) 

Tested on Windows 10, i7-6700HQ @ 2.60 GHz 16Gb, GTX 970m 6Gb (Personal Laptop) 

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

