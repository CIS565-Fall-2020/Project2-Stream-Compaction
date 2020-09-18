#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void addPrev(int n, int *idata, int *odata, int d) {
          int idx = threadIdx.x + (blockIdx.x * blockDim.x);
          int base = 2 << (d - 1);
          if (base + idx >= n) return;
          odata[base + idx] = idata[base + idx] + idata[idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata, *dev_odata;
            cudaMalloc((void **) &dev_idata, n * sizeof(int));
            cudaMalloc((void **) &dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            int iterations = ilog2ceil(n);
            
            for (int d = 1; d <= iterations; d++) {
              int base = 2 << (d - 1);
              int numThreads = n - base;
              dim3 blocks((numThreads + blockSize - 1) / blockSize);
              addPrev<<<blocks, blockSize>>>(n, dev_idata, dev_odata, d);
              std::swap(dev_idata, dev_odata);
            }

            timer().endGpuTimer();
            cudaMemcpy(odata, (iterations % 2 == 0) ? dev_odata : dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
