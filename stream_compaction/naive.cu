#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

int* dev_idata;
int* dev_odata;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kern_NaiveScan(int n, int* odata, int* idata, int pow) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (idx >= pow) {
                odata[idx] = idata[idx - pow] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
        }

        __global__ void kern_Exclusive(int n, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (idx == 0) {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
           

            int blocks = ceil((float)n / (float)blockSize);
            
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            int logVal = ilog2ceil(n);
            for (int d = 1; d <= logVal; d++) {
                kern_NaiveScan <<<blocks, blockSize >>> (n, dev_odata, dev_idata, (int)powf(2, d - 1));
                if (d < logVal) {
                    int* tempPtr = dev_odata;
                    dev_odata = dev_idata;
                    dev_idata = tempPtr;
                }
            }
            kern_Exclusive <<<blocks, blockSize >>> (n, dev_idata, dev_odata);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
