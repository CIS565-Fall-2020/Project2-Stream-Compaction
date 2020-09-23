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
          if (idx >= n) return;
          int base = 1 << (d - 1);
          odata[idx] = idx >= base ? idata[idx - base] + idata[idx] : idata[idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata, *dev_odata;
            cudaMalloc((void **) &dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void **) &dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");

            timer().startGpuTimer();
            int iterations = ilog2ceil(n);
            
            dim3 blocks((n + blockSize - 1) / blockSize);
            for (int d = 1; d <= iterations; d++) {
                if (d % 2 == 1) {
                    addPrev << <blocks, blockSize >> > (n, dev_idata, dev_odata, d);
                }
                else {
                    addPrev << <blocks, blockSize >> > (n, dev_odata, dev_idata, d);
                }
              checkCUDAError("addPrev failed");
            }

            timer().endGpuTimer();
            odata[0] = 0;
            cudaMemcpy(odata + 1, (iterations % 2 == 1) ? dev_odata : dev_idata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
