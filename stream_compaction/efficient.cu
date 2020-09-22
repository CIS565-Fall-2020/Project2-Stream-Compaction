#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_buffer;
        int* dev_booleanBuffer;
        int* dev_scanBuffer;
        int* dev_idata;
        int* dev_odata;

        __global__ void kernUpSweep(int N, int offset, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) {
                return;
            }
            //int offset = powf(2, d);
            if (index % (offset * 2) == 0) {
                data[index + offset * 2 - 1] += data[index + offset - 1];
            }
        }

        __global__ void kernDownSweep(int N, int offset, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) {
                return;
            }
            //int offset = powf(2, d);
            if (index % (offset * 2) == 0) {
                int t = data[index + offset - 1];
                data[index + offset - 1] = data[index + offset * 2 - 1];
                data[index + offset * 2 - 1] += t;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int dmax = ilog2ceil(n);
            int numObjects = powf(2, dmax);
            cudaMalloc((void**)&dev_buffer, numObjects * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 256;
            dim3 numBlocks((numObjects + blockSize - 1) / blockSize);
            
            timer().startGpuTimer();

            for (int i = 0; i < dmax; i++) {
                kernUpSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_buffer);
            }
            cudaMemset(dev_buffer + numObjects - 1, 0, sizeof(int));
            for (int i = dmax - 1; i >= 0; i--) {
                kernDownSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_buffer);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buffer);
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
            int dmax = ilog2ceil(n);
            int numObjects = powf(2, dmax);

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_booleanBuffer, n * sizeof(int));
            cudaMalloc((void**)&dev_scanBuffer, numObjects * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            const int blockSize = 64;

            dim3 numBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            Common::kernMapToBoolean << <numBlocks, blockSize >> > (n, dev_booleanBuffer, dev_idata);

            cudaMemcpy(dev_scanBuffer, dev_booleanBuffer, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int i = 0; i < dmax; i++) {
                kernUpSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_scanBuffer);
            }

            cudaMemset(dev_scanBuffer + numObjects - 1, 0, sizeof(int));
            for (int i = dmax - 1; i >= 0; i--) {
                kernDownSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_scanBuffer);
            }

            int size = 0;
            cudaMemcpy(&size, dev_scanBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            Common::kernScatter << <numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_booleanBuffer, dev_scanBuffer);

            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_booleanBuffer);
            cudaFree(dev_scanBuffer);
            
            return idata[n - 1] ? size + 1 : size;
        }
    }
}
