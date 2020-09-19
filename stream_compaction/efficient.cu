#include "efficient.h"

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        constexpr int log2BlockSize = 9; // 512
        constexpr int blockSize = 1 << log2BlockSize;

        constexpr int numBanks = 32;
        /*__host__ __device__ int _transformIndex(int i) {

        }*/

        __global__ void kernScanPerBlock(int *data, int *lastData) {
            extern __shared__ int buffer[];
            data += blockIdx.x * blockDim.x;

            // copy data to shared memory
            buffer[threadIdx.x] = data[threadIdx.x];
            __syncthreads();

            int lastElem = 0;
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastElem = buffer[threadIdx.x];
            }

            // upward pass
            for (int gap = 2; gap < blockDim.x; gap <<= 1) {
                if ((threadIdx.x & (gap - 1)) == gap - 1) {
                    buffer[threadIdx.x] += buffer[threadIdx.x - (gap >> 1)];
                }
                __syncthreads();
            }

            if (threadIdx.x == blockDim.x - 1) {
                int halfIdx = threadIdx.x >> 1;
                buffer[threadIdx.x] = buffer[halfIdx];
                buffer[halfIdx] = 0;
            }
            __syncthreads();

            // downward pass
            for (int gap = blockDim.x >> 1; gap > 1; gap >>= 1) {
                if ((threadIdx.x & (gap - 1)) == gap - 1) {
                    int prevIdx = threadIdx.x - (gap >> 1);
                    int sum = buffer[threadIdx.x] + buffer[prevIdx];
                    buffer[prevIdx] = buffer[threadIdx.x];
                    buffer[threadIdx.x] = sum;
                }
                __syncthreads();
            }

            // copy data back
            data[threadIdx.x] = buffer[threadIdx.x];
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastData[blockIdx.x] = lastElem + buffer[threadIdx.x];
            }
        }

        __global__ void kernAddConstantToBlock(int *data, const int *amount) {
            data[blockIdx.x * blockDim.x + threadIdx.x] += amount[blockIdx.x];
        }

        void _computeSizes(int n, int log2BlockSize, int *numBlocks, int *bufferSize) {
            *numBlocks = n >> log2BlockSize;
            if ((n & ((1 << log2BlockSize) - 1)) != 0) {
                ++*numBlocks;
            }
            *bufferSize = *numBlocks << log2BlockSize;
        }

        void dev_scan(int n, int *dev_data) {
            assert((n & (blockSize - 1)) == 0);

            if (n > blockSize) {
                int numBlocks = n >> log2BlockSize, numIndirectBlocks, indirectSize;
                _computeSizes(numBlocks, log2BlockSize, &numIndirectBlocks, &indirectSize);

                int *buffer;
                cudaMalloc(&buffer, sizeof(int) * indirectSize);

                kernScanPerBlock<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(dev_data, buffer);
                dev_scan(indirectSize, buffer);
                kernAddConstantToBlock<<<numBlocks, blockSize>>>(dev_data, buffer);

                cudaFree(buffer);
            } else { // just scan the block
                kernScanPerBlock<<<1, blockSize, blockSize * sizeof(int)>>>(dev_data, nullptr);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &numBlocks, &bufferSize);

            int *buffer;
            cudaMalloc(&buffer, sizeof(int) * bufferSize);

            cudaMemcpy(buffer, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            // if integer overflow on the GPU were well-defined we would be able to get away without zeroing the rest
            cudaMemset(buffer + n, 0, sizeof(int) * (bufferSize - n));

            timer().startGpuTimer();
            dev_scan(bufferSize, buffer);
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata, buffer, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(buffer);
            checkCUDAError("efficient scan");
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
            constexpr int log2BlockSize = 9; // block_size = 512

            int numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &numBlocks, &bufferSize);

            int *data, *accum, *out;
            cudaMalloc(&data, sizeof(int) * bufferSize);
            cudaMalloc(&accum, sizeof(int) * bufferSize);
            cudaMalloc(&out, sizeof(int) * bufferSize);

            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(data + n, 0, sizeof(int) * (bufferSize - n));

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, accum, data);
            dev_scan(bufferSize, accum);
            Common::kernScatter<<<numBlocks, blockSize>>>(n, out, data, data, accum);
            timer().endGpuTimer();

            int res;
            cudaMemcpy(odata, out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(&res, accum + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("efficient compaction");
            if (idata[n - 1] != 0) {
                ++res;
            }
            return res;
        }
    }
}
