#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void naiveScanParallel(int n, int startIndex, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index < startIndex) {
                odata[index] = idata[index];
            }
            else {
                odata[index] = idata[index - startIndex] + idata[index];
            }
        }

        // Convert inclusive to exclusive
        __global__ void convert(int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_arr1;
            int* dev_arr2;
            int direction = 1;

            cudaMalloc((void**)&dev_arr1, n * sizeof(int));
            checkCUDAError("dev_arrr1 failed!");
            cudaMalloc((void**)&dev_arr2, n * sizeof(int));
            checkCUDAError("dev_arrr2 failed!");

            cudaMemcpy(dev_arr1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(n); d++) {
                int startIndex = pow(2, d - 1);
                if (direction == 1) {
                    naiveScanParallel << <fullBlocksPerGrid, threadsPerBlock >> > (n, startIndex, dev_arr1, dev_arr2);
                }
                else {
                    naiveScanParallel << <fullBlocksPerGrid, threadsPerBlock >> > (n, startIndex, dev_arr2, dev_arr1);
                }
                direction *= -1;
            }
            convert << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_arr1, dev_arr2);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_arr2, n * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
        }
    }
}
