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

        constexpr int log2BlockSize = 7;
        constexpr int block_size = 1 << log2BlockSize;

        __global__ void kernScanPass(int n, int diff, int *odata, const int *idata) {
            int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
            if (iSelf > n) {
                return;
            }
            if (iSelf >= diff) {
                odata[iSelf] = idata[iSelf] + idata[iSelf - diff];
            } else {
                odata[iSelf] = idata[iSelf];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *buf1 = nullptr, *buf2 = nullptr;
            cudaMalloc(&buf1, sizeof(int) * n);
            cudaMalloc(&buf2, sizeof(int) * n);

            cudaMemcpy(buf1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            {
                timer().startGpuTimer();

                int num_blocks = (n + block_size - 1) / block_size;
                for (int diff = 1; diff < n; diff *= 2) {
                    kernScanPass<<<num_blocks, block_size>>>(n, diff, buf2, buf1);
                    std::swap(buf1, buf2);
                }

                timer().endGpuTimer();
            }

            odata[0] = 0;
            cudaMemcpy(odata + 1, buf1, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

            cudaFree(buf1);
            cudaFree(buf2);
        }
    }
}
