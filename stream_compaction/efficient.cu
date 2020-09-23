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

        // this many elements are processed by one block. cuda block size is half this
        /*constexpr int log2BlockSize = 10; // 1024*/
        /*constexpr int log2BlockSize = 9; // 512*/
        /*constexpr int log2BlockSize = 8; // 256*/
        constexpr int log2BlockSize = 7; // 128
        /*constexpr int log2BlockSize = 6; // 64*/
        constexpr int blockSize = 1 << log2BlockSize;

        constexpr int log2BankSize = 5;

        __device__ int conflictFreeIndex(int i) {
            return i + (i >> log2BankSize);
        }

        __global__ void kernScanPerBlock(int *data, int *lastData) {
            extern __shared__ int buffer[];
            data += blockIdx.x * blockDim.x * 2;

            int offset1 = conflictFreeIndex(threadIdx.x), offset2 = conflictFreeIndex(threadIdx.x + blockDim.x);

            // copy data to shared memory
            buffer[offset1] = data[threadIdx.x];
            buffer[offset2] = data[threadIdx.x + blockDim.x];

            int lastElem = 0;
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastElem = buffer[offset2];
            }
            __syncthreads();

            // upward pass
            for (int halfGap = 1; halfGap < blockDim.x; halfGap <<= 1) {
                if (threadIdx.x < blockDim.x / halfGap) {
                    int
                        id1 = conflictFreeIndex((threadIdx.x * 2 + 1) * halfGap - 1),
                        id2 = conflictFreeIndex((threadIdx.x * 2 + 2) * halfGap - 1);
                    buffer[id2] += buffer[id1];
                }
                __syncthreads();
            }

            if (threadIdx.x == blockDim.x - 1) {
                buffer[conflictFreeIndex(blockDim.x * 2 - 1)] = buffer[offset1];
                buffer[offset1] = 0;
            }
            __syncthreads();

            // downward pass
            for (int halfGap = blockDim.x >> 1; halfGap >= 1; halfGap >>= 1) {
                if (threadIdx.x < blockDim.x / halfGap) {
                    int prevIdx = (threadIdx.x * 2 + 1) * halfGap - 1;
                    int thisIdx = prevIdx + halfGap;
                    prevIdx = conflictFreeIndex(prevIdx);
                    thisIdx = conflictFreeIndex(thisIdx);
                    int sum = buffer[thisIdx] + buffer[prevIdx];
                    buffer[prevIdx] = buffer[thisIdx];
                    buffer[thisIdx] = sum;
                }
                __syncthreads();
            }

            // copy data back
            data[threadIdx.x] = buffer[offset1];
            data[threadIdx.x + blockDim.x] = buffer[offset2];
            if (lastData && threadIdx.x == blockDim.x - 1) {
                lastData[blockIdx.x] = lastElem + buffer[offset2];
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

                kernScanPerBlock<<<
                    numBlocks, blockSize / 2, (blockSize + (blockSize >> log2BankSize)) * sizeof(int)
                >>>(dev_data, buffer);
                dev_scan(indirectSize, buffer);
                kernAddConstantToBlock<<<numBlocks, blockSize>>>(dev_data, buffer);

                cudaFree(buffer);
            } else { // just scan the block
                kernScanPerBlock<<<
                    1, blockSize / 2, (blockSize + (blockSize >> log2BankSize)) * sizeof(int)
                >>>(dev_data, nullptr);
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
            constexpr int log2ScatterBlockSize = 6;
            constexpr int scatterBlockSize = 1 << log2ScatterBlockSize;

            int numBlocks, bufferSize;
            _computeSizes(n, log2BlockSize, &numBlocks, &bufferSize);
            int numScatterBlocks = (n + scatterBlockSize - 1) >> log2ScatterBlockSize;

            int *data, *accum, *out;
            cudaMalloc(&data, sizeof(int) * bufferSize);
            cudaMalloc(&accum, sizeof(int) * bufferSize);
            cudaMalloc(&out, sizeof(int) * bufferSize);

            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, accum, data);
            dev_scan(bufferSize, accum);
            Common::kernScatter<<<numScatterBlocks, (1 << log2ScatterBlockSize)>>>(n, out, data, data, accum);
            timer().endGpuTimer();

            int last = idata[n - 1] != 0 ? 1 : 0, res;
            cudaMemcpy(&res, accum + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            res += last;
            cudaMemcpy(odata, out, sizeof(int) * res, cudaMemcpyDeviceToHost);
            checkCUDAError("efficient compaction");

            cudaFree(data);
            cudaFree(accum);
            cudaFree(out);

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

            timer().startGpuTimer();
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
            timer().endGpuTimer();

            cudaMemcpy(odata, data1, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(data1);
            cudaFree(data2);
            cudaFree(trues);
            cudaFree(falses);
            checkCUDAError("radix sort");
        }
    }
}
