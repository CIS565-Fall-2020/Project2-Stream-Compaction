#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <cstdio>

#define blockSize 128

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int *odata, int *idata, int log2n) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int pow2 = 1;
            /*for (int d = 1; d < log2n; d++) {
                if (index >= pow2) {
                    odata[index] = idata[index - pow2] + idata[index];
                }
                else {
                    odata[index] = idata[index];
                }
                __syncthreads();

                pow2 *= 2;
            }*/
            odata[index] = idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int numBlocks = ceil(n / blockSize);
            const int size = n;
            int* idataCopy = new int[size];
            int i = idata[0];
            cudaMemcpy(idataCopy, idata, sizeof(int) * n, cudaMemcpyHostToHost);
            int j = idataCopy[0];
            kernNaiveScan <<<numBlocks, blockSize>>> (n, odata, idataCopy, ilog2ceil(n));
            timer().endGpuTimer();
        }
    }
}
