#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <iostream>

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }

        __global__ void kernUpSweep(int n, int stepSize, int* data) {
            int idx = (threadIdx.x + blockIdx.x * blockDim.x + 1) * stepSize - 1;
            if (idx >= n) { return; }
            int left_child = idx - (stepSize / 2);
            data[idx] += data[left_child];
        }

        __global__ void kernDownSweep(int n, int stepSize, int* data) {
            int idx = (threadIdx.x + blockIdx.x * blockDim.x + 1) * stepSize - 1;
            if (idx >= n) { return; }
            int left_parent = idx - (stepSize / 2);
            int left_parent_val = data[left_parent];
            int right_parent_val = data[idx];
            data[left_parent] = right_parent_val;
            data[idx] = right_parent_val + left_parent_val;
        }

        __global__ void kernZeroFinalIdx(int n, int* data) {
            data[n - 1] = 0;
        }

        void cudaScan(int n, int* dev_data) {
            // upsweep
            int maxStepSize = 0;
            for (int i = 1; i < n; i = i << 1) {
                int stepSize = i * 2;
                dim3 numBlocks = dim3((int(std::ceil(float(n) / float(stepSize))) + blockSize - 1) / blockSize);
                kernUpSweep << <numBlocks, blockSize >> > (n, stepSize, dev_data);
                maxStepSize = i;
            }

            //set final idx = 0
            kernZeroFinalIdx << <1, 1 >> > (n, dev_data);

            // downsweep
            for (int i = maxStepSize; i >= 1; i = i >> 1) {
                int stepSize = i * 2;
                dim3 numBlocks = dim3((int(std::ceil(float(n) / float(stepSize))) + blockSize - 1) / blockSize);
                kernDownSweep << <numBlocks, blockSize >> > (n, stepSize, dev_data);
            }

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int padded_length = 1 << int(std::ceil(std::log2(n)));
            int* dev_data;
            cudaMalloc((void**)&dev_data, padded_length * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            cudaScan(padded_length, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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

            int padded_length = 1 << int(std::ceil(std::log2(n)));
            int* dev_in;
            int* dev_out;
            int* dev_bools;
            cudaMalloc((void**)&dev_in, padded_length * sizeof(int));
            cudaMalloc((void**)&dev_out, padded_length * sizeof(int));
            cudaMalloc((void**)&dev_bools, padded_length * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 nBlocks(((n + blockSize - 1) / blockSize));
            Common::kernMapToBoolean<<<nBlocks, blockSize>>>(n, dev_bools, dev_in);
            cudaScan(padded_length, dev_bools);
            Common::kernScatter<<<nBlocks, blockSize>>>(n, dev_out, dev_in, dev_bools);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            int total_count;
            cudaMemcpy(&total_count, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_bools);
            cudaFree(dev_in);

            // handle case where last number is not 0
            if (idata[n - 1] != 0) {
                odata[total_count] = idata[n - 1];
                total_count += 1;
            }

            return total_count;
        }
    }
}
