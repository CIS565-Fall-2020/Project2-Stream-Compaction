#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


// General globals
int* dev_data;
int* dev_oData;

// Scan and Compact
int* dev_scanData;
int* dev_boolData;

// Radix sort
int* dev_bData;
int* dev_eData;
int* dev_fData;
int* dev_tData;
int* dev_dData;
int* dev_totalFalses;

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

            kernSetRootNode << <1, 1 >> > (size, dev_boolData);

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


    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // bit number goes from 1 to n
        __global__ void kernComputeBArray(int n, int bitNumber, int* bdata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int data = idata[index];
            int bit = (data & (1 << bitNumber - 1)) != 0;
            bdata[index] = bit;
        }

        __global__ void kernComputeEArray(int n, int* edata, int* fdata, const int* bdata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            edata[index] = !(bdata[index]);
            fdata[index] = edata[index];
        }

        __global__ void kernComputeTotalFalses(int n, int* out, int* edata, int* fdata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            *out = edata[n - 1] + fdata[n - 1];
        }

        __global__ void kernComputeTArray(int n, int* tdata, int* edata, int* fdata, int* totalFalses) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            tdata[index] = index - fdata[index] + *totalFalses;
        }

        __global__ void kernComputeDArray(int n, int* ddata, int* bdata, int* fdata, int* tdata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            ddata[index] = bdata[index] ? tdata[index] : fdata[index];
        }

        __global__ void kernScatterRadix(int n, int* odata, int* ddata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            odata[ddata[index]] = idata[index];
        }

        void radixSort(int n, int* odata, const int* idata) {
            int numBlocks = ceil((float)n / (float)blockSize);
            int log2n = ilog2ceil(n);
            const int size = (int)powf(2, log2n);

            cudaMalloc((void**)&dev_data, sizeof(int) * n);
            cudaMalloc((void**)&dev_bData, sizeof(int) * n);
            cudaMalloc((void**)&dev_eData, sizeof(int) * n);
            cudaMalloc((void**)&dev_fData, sizeof(int) * size);
            cudaMalloc((void**)&dev_tData, sizeof(int) * n);
            cudaMalloc((void**)&dev_dData, sizeof(int) * n);
            cudaMalloc((void**)&dev_totalFalses, sizeof(int));


            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // Find max element.
            int maxElement = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] > maxElement)
                    maxElement = idata[i];
            }

            for (int bitNumber = 1; maxElement /((int) powf(2, bitNumber - 1)) > 0; bitNumber++) {
                kernComputeBArray << <numBlocks, blockSize >> > (n, bitNumber, dev_bData, dev_data);
                kernComputeEArray << <numBlocks, blockSize >> > (n, dev_eData, dev_fData, dev_bData);


                // Compute f Array
                for (int d = 0; d <= log2n - 1; d++) {
                    StreamCompaction::Efficient::kernStepUpSweep << <numBlocks, blockSize >> > (size, dev_fData, (int)powf(2, d));
                }

                StreamCompaction::Efficient::kernSetRootNode << <1, 1 >> > (size, dev_fData);

                for (int d = log2n - 1; d >= 0; d--) {
                    StreamCompaction::Efficient::kernStepDownSweep << <numBlocks, blockSize >> > (size, dev_fData, (int)powf(2, d));
                }

                kernComputeTotalFalses << <1, 1 >> > (n, dev_totalFalses, dev_eData, dev_fData);
                kernComputeTArray << <numBlocks, blockSize >> > (n, dev_tData, dev_eData, dev_fData, dev_totalFalses);
                kernComputeDArray << <numBlocks, blockSize >> > (n, dev_dData, dev_bData, dev_fData, dev_tData);
                kernScatterRadix << <numBlocks, blockSize >> > (n, dev_data, dev_dData, dev_data);
            }

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(dev_bData);
            cudaFree(dev_eData);
            cudaFree(dev_fData);
            cudaFree(dev_tData);
            cudaFree(dev_dData);
            cudaFree(dev_totalFalses);
            
        }
     }

}
