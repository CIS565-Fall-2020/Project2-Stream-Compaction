#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <cstdio>

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

int *dev_idata;
int* dev_odata;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernStepNaiveScan(int n, int *odata, int *idata, int pow2) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index >= pow2) {
                odata[index] = idata[index - pow2] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernMakeExclusive(int n, int *odata, int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numBlocks = ceil((float)n / (float)blockSize);
            const int size = n;
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            int log2n = ilog2ceil(n);
            for (int d = 1; d <= log2n; d++) {
                kernStepNaiveScan << <numBlocks, blockSize >> > (n, dev_odata, dev_idata, (int) powf(2, d - 1));
                if (d < log2n) {
                    int* tempPtr = dev_odata;
                    dev_odata = dev_idata;
                    dev_idata = tempPtr;
                }
            }

            // The correct data will be in odata, now have to make exclusive and store
            // in idata, contrary to the original name's intention
            kernMakeExclusive << <numBlocks, blockSize >> > (n, dev_idata, dev_odata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
