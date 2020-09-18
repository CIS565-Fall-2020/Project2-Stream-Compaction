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

        __global__ void kernNaiveScan(int n, int offset,
            int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= offset && index < n) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else if (index < offset) {
                odata[index] = idata[index];
            }
        }

        __global__ void kernRightShift(int n, int* odata, int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index == 0) {
                odata[index] = 0;
            }
            else if (index < n) {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) { // TODO
            // Two ping pong buffers
            int* dev_data1 = nullptr;
            int* dev_data2 = nullptr;

            // Allocate memory on device
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data1 failed!");
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data2 failed!");

            // Copy data from host to device
            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            if (n > 0) {
                dim3 threadsPerBlock(128);
                int d = ilog2ceil(n);
                int offset = 1;
                dim3 blocks(n / threadsPerBlock.x + 1);
                for (int i = 1; i <= d; i++) {
                    kernNaiveScan << <blocks, threadsPerBlock >> >
                        (n, offset, dev_data2, dev_data1);
                    std::swap(dev_data1, dev_data2);
                    offset <<= 1;
                }

                // Right shift to get the exclusive prefix sum
                kernRightShift << <blocks, threadsPerBlock >> >
                    (n, dev_data2, dev_data1);
            }

            timer().endGpuTimer();

            // Copy data back from device to host
            cudaMemcpy(odata, dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Free device memory
            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
