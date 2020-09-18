#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

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

        void _computeSizes(int n, int log2BlockSize, int *blockSize, int *numBlocks, int *bufferSize) {
            *blockSize = 1 << log2BlockSize;
            *numBlocks = n >> log2BlockSize;
            if ((n & (*blockSize - 1)) != 0) {
                ++*numBlocks;
            }
            *bufferSize = *numBlocks << log2BlockSize;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int log2BlockSize = 9; // block_size = 512

            int blockSize, numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &blockSize, &numBlocks, &bufferSize);

            int *buffer;
            cudaMalloc(&buffer, sizeof(int) * bufferSize);

            cudaMemcpy(buffer, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            // if integer overflow on the GPU were well-defined we would be able to get away without zeroing the rest
            cudaMemset(buffer + n, 0, sizeof(int) * (bufferSize - n));

            timer().startGpuTimer();
            kernScanPerBlock<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(buffer, nullptr);
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata, buffer, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(buffer);
        }

        __global__ void kernConvertToBinary(int n, int *odata, const int *idata) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            if (iSelf >= n) {
                return;
            }
            odata[iSelf] = idata[iSelf] != 0 ? 1 : 0;
        }

        __global__ void kernCompact(int n, int *out, const int *data, const int *accum) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            if (iSelf >= n) {
                return;
            }

            int val = data[iSelf];
            if (val != 0) {
                out[accum[iSelf]] = val;
            }
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
            int log2BlockSize = 9; // block_size = 512

            int blockSize, numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &blockSize, &numBlocks, &bufferSize);

            int *data, *accum, *out;
            cudaMalloc(&data, sizeof(int) * bufferSize);
            cudaMalloc(&accum, sizeof(int) * bufferSize);
            cudaMalloc(&out, sizeof(int) * bufferSize);

            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(data + n, 0, sizeof(int) * (bufferSize - n));

            timer().startGpuTimer();
            kernConvertToBinary<<<numBlocks, blockSize>>>(n, accum, data);
            kernScanPerBlock<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(accum, nullptr);
            kernCompact<<<numBlocks, blockSize>>>(n, out, data, accum);
            timer().endGpuTimer();

            int res;
            cudaMemcpy(odata, out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(&res, accum + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] != 0) {
                ++res;
            }
            return res;
        }
    }
}
