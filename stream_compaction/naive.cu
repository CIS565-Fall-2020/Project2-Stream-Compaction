#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void addition_process(int* dev_array1, int* dev_array2, const int d, const int n) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            // int two_power_d_min_1 = powf(2.0, d - 1);
            int two_power_d_min_1 = 1 << (d - 1);
            
            if (index >= two_power_d_min_1) {
                dev_array2[index] = dev_array1[index - two_power_d_min_1] + dev_array1[index];
            }
            else {
                dev_array2[index] = dev_array1[index];
            }
        }

        __global__ void right_shift(int* dev_array1, int* dev_array2, const int n) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index == 0) {
                dev_array2[index] = 0;
            }
            else {
                dev_array2[index] = dev_array1[index - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data_array1, * dev_data_array2;
            // Init all the requirement data:
            int sizeInBytes = n * sizeof(int);
            int blockSize = 32;
            dim3 fullBlocksPerGrid((n / blockSize) + 1);

            cudaMalloc((void**)&dev_data_array1, sizeInBytes);
            checkCUDAError("cudaMalloc dev_data_array1 failed!");
            cudaMalloc((void**)&dev_data_array2, sizeInBytes);
            checkCUDAError("cudaMalloc dev_data_array2 failed!");
            cudaMemcpy(dev_data_array1, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data_array1 failed!");
            cudaMemcpy(dev_data_array2, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data_array2 failed!");

            timer().startGpuTimer();
            // TODO
            // Inclusive scan:
            int d_max = ilog2ceil(n);
            for (int d = 1; d <= d_max; ++d)
            {
                addition_process <<< fullBlocksPerGrid, blockSize >>> (dev_data_array1, dev_data_array2, d, n);
                checkCUDAError("Naive addition_process failed!");
                int* temp = dev_data_array1;
                dev_data_array1 = dev_data_array2;
                dev_data_array2 = temp;
            }
            // Right shift to get an exclusive scan:
            right_shift <<< fullBlocksPerGrid, blockSize >>> (dev_data_array1, dev_data_array2, n);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data_array2, sizeInBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_data_array1);
            checkCUDAError("cudaFree(dev_data_array1) failed!");
            cudaFree(dev_data_array2);
            checkCUDAError("cudaFree(dev_data_array2) failed!");
        }
    }
}
