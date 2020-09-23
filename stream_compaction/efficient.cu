#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

int* dev_data;
int* dev_oData;
int* dev_scanData;
int* dev_boolData;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

       
        
        __global__ void kern_UpSweep(int n, int* arr, int pow) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (index >= n) {
                return;
            }

            if (index % (2 * pow) == 0) {
                arr[index + 2 * pow - 1] += arr[index + pow - 1];
            }

        }

        __global__ void kern_SetRoot(int n, int* arr) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            arr[n - 1] = 0;
        }

        __global__ void kern_DownSweep(int n, int* arr, int pow) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index % (2 * pow) == 0) {
                int temp = arr[index + pow - 1];
                arr[index + pow - 1] = arr[index + 2 * pow - 1];
                arr[index + 2 * pow - 1] += temp;
            }
        }

        /**
        * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        */
        void scan(int n, int* odata, const int* idata) {
            int blocks = ceil((float)n / (float)blockSize);
            int logN = ilog2ceil(n);
            const int len = (int)powf(2, logN);

            cudaMalloc((void**)&dev_data, sizeof(int) * (int)powf(2, logN));
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            for (int d = 0; d <= logN - 1; d++) {
                kern_UpSweep << <blocks, blockSize >> > (len, dev_data, (int)powf(2, d));
            }

            kern_SetRoot << <1, 1 >> > (len, dev_data);

            for (int d = logN - 1; d >= 0; d--) {
                kern_DownSweep << <blocks, blockSize >> > (len, dev_data, (int)powf(2, d));
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
            int logN = ilog2ceil(n);
            const int len = (int)powf(2, logN);

            cudaMalloc((void**)&dev_data, sizeof(int) * len);
            cudaMalloc((void**)&dev_boolData, sizeof(int) * len);
            cudaMalloc((void**)&dev_oData, sizeof(int) * n);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            timer().startGpuTimer();

            // TODO -> DONE
            StreamCompaction::Common::kernMapToBoolean << <numBlocks, blockSize >> > (len, dev_boolData, dev_data);

            for (int d = 0; d <= logN - 1; d++) {
                kern_UpSweep << <numBlocks, blockSize >> > (len, dev_boolData, (int)powf(2, d));
            }

            kern_SetRoot << <1, 1 >> > (len, dev_boolData);

            for (int d = logN - 1; d >= 0; d--) {
                kern_DownSweep << <numBlocks, blockSize >> > (len, dev_boolData, (int)powf(2, d));
            }

            StreamCompaction::Common::kernScatter << <numBlocks, blockSize >> > (n, dev_oData, dev_data, dev_boolData, nullptr);

            timer().endGpuTimer();

            int* finalBoolArr = new int[n];
            cudaMemcpy(odata, dev_oData, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(finalBoolArr, dev_boolData, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(dev_boolData);
            cudaFree(dev_oData);

            if (idata[n - 1] == 0) {
                return finalBoolArr[n - 1];
            }

            return finalBoolArr[n - 1] + 1;
        }
    }
}
