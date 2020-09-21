#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

int* dev_data;
int* dev_scanData;
int* dev_boolData;
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

        __global__ void kernSetRootNode(int n, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            data[n - 1] = 0;
        }

        // if you copy the NPOT array into a POT GPU bfffer, and you pad any of the data
        // that doesn't exist with zeroes, it makes result of the scan instead of the
        // array being like. if you pad zeroes, the value gets repeated a bunch of times

        // could also try to adjust how many threads get run

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

            cudaMalloc((void**)&dev_data, sizeof(int) * size);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int d = 0; d <= log2n - 1; d++) {
                kernStepUpSweep <<<numBlocks, blockSize >>> (size, dev_data, (int)powf(2, d));
            }

            kernSetRootNode << <1, 1 >> > (size, dev_data);
            
            for (int d = log2n - 1; d >= 0; d--) {
                kernStepDownSweep << <numBlocks, blockSize >> > (size, dev_data, (int)powf(2, d));
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

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
            int numBlocks = ceil((float)n / (float)blockSize);
            int log2n = ilog2ceil(n);
            const int size = (int)powf(2, log2n);

            cudaMalloc((void**)&dev_data, sizeof(int) * size);
            cudaMalloc((void**)&dev_boolData, sizeof(int) * size);
            cudaMalloc((void**)&dev_oData, sizeof(int) * n);

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Make temporary array
            StreamCompaction::Common::kernMapToBoolean << <numBlocks, blockSize >> > (size, dev_boolData, dev_data);

            // Scan
            for (int d = 0; d <= log2n - 1; d++) {
                kernStepUpSweep << <numBlocks, blockSize >> > (size, dev_boolData, (int)powf(2, d));
            }

            odata[size - 1] = 0;
            cudaMemcpy(dev_boolData + size - 1, odata + size - 1, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = log2n - 1; d >= 0; d--) {
                kernStepDownSweep << <numBlocks, blockSize >> > (size, dev_boolData, (int)powf(2, d));
            }

            StreamCompaction::Common::kernScatter << <numBlocks, blockSize >> > (n, dev_oData, dev_data, dev_boolData, nullptr);

            timer().endGpuTimer();

            int* boolArray = new int[n];
            cudaMemcpy(odata, dev_oData, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(boolArray, dev_boolData, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(dev_boolData);
            cudaFree(dev_oData);

            if (idata[n - 1] == 0) {
                return boolArray[n - 1];
            }

            return boolArray[n - 1] + 1;
        }
    }
}
