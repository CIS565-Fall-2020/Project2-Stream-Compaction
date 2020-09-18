#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

int* dev_data;
int* dev_scanData;
int* dev_tempData;
int* dev_oData;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernStepUpSweep(int n, int *data, int pow2) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index % (2 * pow2) == 0) {
                data[index + 2 * pow2 - 1] += data[index + pow2 - 1];
            }

        }

        __global__ void kernStepDownSweep(int n, int* data, int pow2) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index % (2 * pow2) == 0) {
                int t = data[index + pow2 - 1];
                data[index + pow2 - 1] = data[index + 2 * pow2 - 1];
                data[index + 2 * pow2 - 1] += t;
            }   
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numBlocks = ceil((float)n / (float)blockSize);
            int log2n = ilog2ceil(n);
            const int size = (int)powf(2, log2n);

            cudaMalloc((void**)&dev_scanData, sizeof(int) * size);
            cudaMemcpy(dev_scanData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int d = 0; d <= log2n - 1; d++) {
                kernStepUpSweep <<<numBlocks, blockSize >>> (size, dev_scanData, (int)powf(2, d));
            }

            odata[size - 1] = 0;
            cudaMemcpy(dev_scanData + size - 1, odata + size - 1, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = log2n - 1; d >= 0; d--) {
                kernStepDownSweep << <numBlocks, blockSize >> > (size, dev_scanData, (int)powf(2, d));
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_scanData, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_scanData);
        }

        __global__ void kernMakeTempArray(int n, int *tempData, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            tempData[index] = data[index] == 0 ? 0 : 1;
        }

        __global__ void kernScatter(int n, int* odata, int* tempData, int* data, int* scanData) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (tempData[index] != 0) {
                odata[scanData[index]] = data[index];
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
            int numBlocks = ceil((float)n / (float)blockSize);
            int log2n = ilog2ceil(n);
            const int size = (int)powf(2, log2n);

            cudaMalloc((void**)&dev_data, sizeof(int) * size);
            cudaMalloc((void**)&dev_scanData, sizeof(int) * n);
            cudaMalloc((void**)&dev_tempData, sizeof(int) * n);
            cudaMalloc((void**)&dev_oData, sizeof(int) * n);

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_scanData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Make temporary array
            kernMakeTempArray << <numBlocks, blockSize >> > (n, dev_tempData, dev_data);

            // Scan
            for (int d = 0; d <= log2n - 1; d++) {
                kernStepUpSweep << <numBlocks, blockSize >> > (n, dev_scanData, (int)powf(2, d));
            }

            odata[n - 1] = 0;

            cudaMemcpy(dev_scanData + n - 1, odata + n - 1, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = log2n - 1; d >= 0; d--) {
                kernStepDownSweep << <numBlocks, blockSize >> > (n, dev_scanData, (int)powf(2, d));
            }

            kernScatter << <numBlocks, blockSize >> > (n, dev_tempData, dev_oData, dev_data, dev_scanData);

            timer().endGpuTimer();
            cudaFree(dev_data);
            cudaFree(dev_tempData);
            cudaFree(dev_scanData);
            cudaFree(dev_oData);

            for (int i = 0; i < n; i++) {
                if (odata[i] == 0) {
                    return i;
                }
            }

            return n;
        }
    }
}
