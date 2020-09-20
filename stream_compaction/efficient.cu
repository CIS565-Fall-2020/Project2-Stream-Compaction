#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /* Copy initial data over and pad 0's if out of scope of initial size
         * aka the input array has a smaller initial size than the final array,
         * and anything larger than index [size of input array] will be 0 in the output array
         */
        __global__ void formatInitData(int initSize, int finalSize, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= initSize && index < finalSize) {
            data[index] = 0;
          }
        }

        __global__ void upSweep(int n, int offset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index < n && index % offset == 0) {
            data[index + offset - 1] = data[index + offset / 2 - 1] + data[index + offset - 1];
          }
        }

        __global__ void downSweep(int n, int offset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index < n && index % offset == 0) {
            int halfOffset = offset / 2; // Helps to find left child
            int t = data[index + halfOffset - 1];
            // Set right child to be the same as parent's value
            data[index + halfOffset - 1] = data[index + offset - 1];
            // Set left child to be the sum of parent and parent's sibling
            data[index + offset - 1] += t;
          }
        }

        __global__ void setLastElementZero(int n, int* data) {
          data[n - 1] = 0;
        }

        __global__ void shiftRight(int n, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          if (index == 0) {
            data[index] = 0;
          }
          else {
            data[index] = data[index - 1];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            // TODO
            if (n < 1) {
              return;
            }
            // Calculate the number of elements the input can be treated as an array with a power of two elements
            int kernelInvokeCount = ilog2ceil(n);
            int n2 = pow(2, kernelInvokeCount);

            int blockSize = 128;
            dim3 blockCount((n2 + blockSize - 1) / blockSize);

            // Declare, allocate, and transfer data on gpu from cpu
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            // Format input data (pad 0s to the closest power of two elements, inclusively)
            formatInitData << <blockCount, blockSize >> > (n, n2, dev_odata);

            for (int i = 0; i <= kernelInvokeCount; i++) {
              int offset = pow(2, i + 1);
              upSweep << <blockCount, blockSize >> > (n2, offset, dev_odata);
            }

            setLastElementZero << <blockCount, blockSize >> > (n2, dev_odata);

            for (int i = kernelInvokeCount - 1; i >= 0; i--) {
              int offset = pow(2, i + 1);
              downSweep << <blockCount, blockSize >> > (n2, offset, dev_odata);
              // Transfer data from gpu to cpu
              cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
              checkCUDAError("cudaMemcpy dev_odata failed!");
            }

            // Transfer data from gpu to cpu
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");

            //timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            // TODO
            
            int blockSize = 128;
            dim3 blockCount((n + blockSize - 1) / blockSize);

            // Declare, allocate memory in GPU and transfer memory from CPU to GPU
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            // Compute temorary arrray containing
            // 1 if corresponding element meets criteria (of not equal to 0)
            // 0 if element does not meete criteria (of not equal to 0)
            StreamCompaction::Common::kernMapToBoolean << <blockCount, blockSize >> > (n, dev_bools, dev_idata);

            // Run exclusive scan on temporary array
            scan(n, dev_indices, dev_bools);
            StreamCompaction::Common::kernScatter << <blockCount, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            std::unique_ptr<int[]> bools{ new int[n] };
            std::unique_ptr<int[]> indices{ new int[n] };

            cudaMemcpy(bools.get(), dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bools failed!");

            cudaMemcpy(indices.get(), dev_indices, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bools failed!");

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            // Clean up
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            int remaining = bools[n - 1] == 1? indices[n - 1] : indices[n - 1] - 1;
            remaining++; 

            timer().endGpuTimer();
            return remaining;
        }
    }
}
