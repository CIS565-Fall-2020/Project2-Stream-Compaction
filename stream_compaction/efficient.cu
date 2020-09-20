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

        // block size when processing. cuda block size is half this
        constexpr int log2BlockSize = 9; // 512
        constexpr int blockSize = 1 << log2BlockSize;

        __global__ void kernScanPerBlock(int *data, int *lastData) {
            extern __shared__ int buffer[];
            data += blockIdx.x * blockDim.x * 2;

            // copy data to shared memory
            buffer[threadIdx.x * 2] = data[threadIdx.x * 2];
            buffer[threadIdx.x * 2 + 1] = data[threadIdx.x * 2 + 1];
            __syncthreads();

            int lastElem = 0;
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastElem = buffer[threadIdx.x * 2 + 1];
            }

            // upward pass
            for (int halfGap = 1; halfGap < blockDim.x; halfGap <<= 1) {
                if (threadIdx.x < blockDim.x / halfGap) {
                    buffer[(threadIdx.x * 2 + 2) * halfGap - 1] += buffer[(threadIdx.x * 2 + 1) * halfGap - 1];
                }
                __syncthreads();
            }

            if (threadIdx.x == blockDim.x - 1) {
                int halfIdx = blockDim.x - 1;
                buffer[blockDim.x * 2 - 1] = buffer[threadIdx.x];
                buffer[threadIdx.x] = 0;
            }
            __syncthreads();

            // downward pass
            for (int halfGap = blockDim.x >> 1; halfGap >= 1; halfGap >>= 1) {
                if (threadIdx.x < blockDim.x / halfGap) {
                    int prevIdx = (threadIdx.x * 2 + 1) * halfGap - 1;
                    int thisIdx = prevIdx + halfGap;
                    int sum = buffer[thisIdx] + buffer[prevIdx];
                    buffer[prevIdx] = buffer[thisIdx];
                    buffer[thisIdx] = sum;
                }
                __syncthreads();
            }

            // copy data back
            data[threadIdx.x * 2] = buffer[threadIdx.x * 2];
            data[threadIdx.x * 2 + 1] = buffer[threadIdx.x * 2 + 1];
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastData[blockIdx.x] = lastElem + buffer[threadIdx.x * 2 + 1];
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

                kernScanPerBlock<<<numBlocks, blockSize / 2, blockSize * sizeof(int)>>>(dev_data, buffer);
                dev_scan(indirectSize, buffer);
                kernAddConstantToBlock<<<numBlocks, blockSize>>>(dev_data, buffer);

                cudaFree(buffer);
            } else { // just scan the block
                kernScanPerBlock<<<1, blockSize / 2, blockSize * sizeof(int)>>>(dev_data, nullptr);
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


        __global__ void kernExtractBit(int n, int bit, int *odata, const int *idata) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            if (iSelf >= n) {
                return;
            }
            odata[iSelf] = (idata[iSelf] & (1 << bit)) != 0 ? 1 : 0;
        }

        __global__ void kernNegate(int *odata, const int *idata) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            odata[iSelf] = idata[iSelf] == 0 ? 1 : 0;
        }

        __global__ void kernRadixSortScatter(
            int n, int numFalses, int bit, int *odata, const int *idata, const int *trues, const int *falses
        ) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            if (iSelf >= n) {
                return;
            }
            int value = idata[iSelf], index;
            if ((value & (1 << bit)) != 0) {
                index = trues[iSelf] + numFalses;
            } else {
                index = falses[iSelf];
            }
            odata[index] = value;
        }

        void radix_sort(int n, int *odata, const int *idata) {
            constexpr int numIntBits = sizeof(int) * 8 - 1;

            int numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &numBlocks, &bufferSize);

            int *data1, *data2, *trues, *falses;
            cudaMalloc(&data1, sizeof(int) * n);
            cudaMalloc(&data2, sizeof(int) * n);
            cudaMalloc(&trues, sizeof(int) * bufferSize);
            cudaMalloc(&falses, sizeof(int) * bufferSize);

            cudaMemcpy(data1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            for (int i = 0; i < numIntBits; ++i) {
                kernExtractBit<<<numBlocks, blockSize>>>(n, i, trues, data1);
                kernNegate<<<numBlocks, blockSize>>>(falses, trues);
                dev_scan(bufferSize, trues);
                dev_scan(bufferSize, falses);
                int numFalses, lastElem;
                cudaMemcpy(&lastElem, data1 + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&numFalses, falses + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                if ((lastElem & (1 << i)) == 0) {
                    ++numFalses;
                }
                kernRadixSortScatter<<<numBlocks, blockSize>>>(n, numFalses, i, data2, data1, trues, falses);
                std::swap(data1, data2);
            }

            cudaMemcpy(odata, data1, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(data1);
            cudaFree(data2);
            cudaFree(trues);
            cudaFree(falses);
            checkCUDAError("radix sort");
        }
    }
}
