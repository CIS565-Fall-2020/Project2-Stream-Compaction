#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <device_launch_parameters.h>
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernNonPaddedToPadded(int nPadded, int n, int* dev_vec_padded, int* dev_vec_nonPadded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < nPadded) {
                if (index < n) {
                    dev_vec_padded[index] = dev_vec_nonPadded[index];
                } else {
                    dev_vec_padded[index] = 0;
                }

            }
        }

        __global__ void kernPaddedToNonPadded(int n, int* dev_vec_padded, int* dev_vec_nonPadded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            //
            if (index < n) {
                dev_vec_nonPadded[index] = dev_vec_padded[index];
            }
        }

        __global__ void kernReduce(int nPadded, int d, int* dev_vec_padded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < nPadded) {
                if (index % (1 << (d + 1)) == 0) { // (int)fmodf(index, 1 << (d + 1))
                    dev_vec_padded[index + (1 << (d + 1)) - 1] += dev_vec_padded[index + (1 << d) - 1];
                }
            }
        }

        __global__ void kernSetRootToZero(int nPadded, int* dev_vec_padded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == nPadded - 1) {
                dev_vec_padded[index] = 0;
            }
        }

        __global__ void downSweep(int nPadded, int d, int* dev_vec_padded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < nPadded) {
                if (index % (1 << (d + 1)) == 0) {
                    int t = dev_vec_padded[index + (1 << d) - 1];
                    dev_vec_padded[index + (1 << d) - 1] = dev_vec_padded[index + (1 << (d + 1)) - 1];
                    dev_vec_padded[index + (1 << (d + 1)) - 1] += t;
                }
            }

        }

        void scan(int n, int *odata, const int *idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int nPadded = n;
            if (paddedSize > n) {
                nPadded = paddedSize;
            }

            int* dev_vec_nonPadded;
            int* dev_vec_padded;
            cudaMalloc((void**)&dev_vec_nonPadded, n * sizeof(int));
            cudaMemcpy(dev_vec_nonPadded, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_vec_padded, nPadded * sizeof(int));

            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGridPadded((nPadded + blockSize - 1) / blockSize);
            // Pad 0s
            kernNonPaddedToPadded << <fullBlocksPerGridPadded, blockSize >> > (nPadded, n, dev_vec_padded, dev_vec_nonPadded);
            
            // Reduce/Up-Sweep
            for (int d = 0; d <= ilog2ceil(nPadded) - 1; d++) {
                kernReduce << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }
            // Set Root To Zero And Down-Sweep
            kernSetRootToZero << <fullBlocksPerGridPadded, blockSize >> > (nPadded, dev_vec_padded);

            for (int d = ilog2ceil(nPadded) - 1; d >= 0; d--) {
                downSweep << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }
            
            // Unpad 0s
            kernPaddedToNonPadded << <fullBlocksPerGrid, blockSize >> > (n, dev_vec_padded, dev_vec_nonPadded);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_vec_nonPadded, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_vec_nonPadded);
            cudaFree(dev_vec_padded);
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

        __global__ void kernMakeBool(int num, int* dev_vec) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < num && dev_vec[index] != 0) {
                dev_vec[index] = 1;
            }
        }

        __global__ void kernScatter(int n, int* dev_idata, int* dev_vec_nonPadded, int* dev_result) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n && dev_idata[index] != 0) {
                dev_result[dev_vec_nonPadded[index]] = dev_idata[index];
            }
        }
        __global__ void kernCheckNonZeroNum(int n, int* dev_result, int* num) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (dev_result[index] == 0 && dev_result[index - 1] != 0) {
                    *num = index;
                }
            }
        }

        int compact(int n, int *odata, const int *idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int nPadded = n;
            if (paddedSize > n) {
                nPadded = paddedSize;
            }

            int* dev_vec_nonPadded;
            int* dev_vec_padded;
            cudaMalloc((void**)&dev_vec_nonPadded, n * sizeof(int));
            cudaMemcpy(dev_vec_nonPadded, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_vec_padded, nPadded * sizeof(int));

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_result;
            cudaMalloc((void**)&dev_result, n * sizeof(int));

            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGridPadded((nPadded + blockSize - 1) / blockSize);
            
            // Pad 0s
            kernNonPaddedToPadded << <fullBlocksPerGridPadded, blockSize >> > (nPadded, n, dev_vec_padded, dev_vec_nonPadded);
            
            // Convert to 0s and 1s
            kernMakeBool << <fullBlocksPerGridPadded, blockSize >> > (nPadded, dev_vec_padded);

            // Reduce/Up-Sweep
            for (int d = 0; d <= ilog2ceil(nPadded) - 1; d++) {
                kernReduce << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }
            // Set Root To Zero And Down-Sweep
            kernSetRootToZero << <fullBlocksPerGridPadded, blockSize >> > (nPadded, dev_vec_padded);

            for (int d = ilog2ceil(nPadded) - 1; d >= 0; d--) {
                downSweep << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }
            
            // Unpad 0s
            kernPaddedToNonPadded << <fullBlocksPerGrid, blockSize >> > (n, dev_vec_padded, dev_vec_nonPadded);

            // Scatter
            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_vec_nonPadded, dev_result);

            int* dev_nonZeroNum;
            cudaMalloc((void**)&dev_nonZeroNum, sizeof(int));
            int nonZeroNum;
            kernCheckNonZeroNum << <fullBlocksPerGrid, blockSize >> > (n, dev_result, dev_nonZeroNum);

            timer().endGpuTimer();

            cudaMemcpy(&nonZeroNum, dev_nonZeroNum, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_result, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_vec_nonPadded);
            cudaFree(dev_vec_padded);
            cudaFree(dev_idata);
            cudaFree(dev_result);
            cudaFree(dev_nonZeroNum);

            std::cout << nonZeroNum;
            return nonZeroNum;
        }
    }
}
