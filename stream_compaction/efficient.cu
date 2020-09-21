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

        int numObjects;
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

        void efficient_scan(int n, int* odata, const int* idata) {
            int dmax = ilog2ceil(n);
            numObjects = powf(2, dmax);
            cudaMalloc((void**)&dev_buffer, numObjects * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 256;
            dim3 numBlocks((numObjects + blockSize - 1) / blockSize);

            for (int i = 0; i < dmax; i++) {
                kernUpSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_buffer);
            }
            cudaMemset(dev_buffer + numObjects - 1, 0, sizeof(int));
            for (int i = dmax - 1; i >= 0; i--) {
                kernDownSweep << <numBlocks, blockSize >> > (numObjects, int(powf(2, i)), dev_buffer);
            }
            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buffer);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            efficient_scan(n, odata, idata);
            timer().endGpuTimer();
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
            timer().startGpuTimer();

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_booleanBuffer, n * sizeof(int));
            cudaMalloc((void**)&dev_scanBuffer, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            const int blockSize = 64;
            numObjects = n;
            dim3 numBoidBlocks((numObjects + blockSize - 1) / blockSize);

            Common::kernMapToBoolean << <numBoidBlocks, blockSize >> > (n, dev_booleanBuffer, dev_idata);
            int* host_boolean = new int[n];
            cudaMemcpy(host_boolean, dev_booleanBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            efficient_scan(n, odata, host_boolean);
            int size = odata[n - 1];
            cudaMemcpy(dev_scanBuffer, odata, n * sizeof(int), cudaMemcpyHostToDevice);
            Common::kernScatter << <numBoidBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_booleanBuffer, dev_scanBuffer);
            
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_booleanBuffer);
            cudaFree(dev_scanBuffer);
            timer().endGpuTimer();
            
            return idata[n - 1] ? size + 1 : size;
        }
    }
}
