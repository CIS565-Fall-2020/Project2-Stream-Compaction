#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>
#include <iostream>
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScan(int n, int bar, int *in, int *out) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            if (k >= bar) {
                out[k] = in[k - bar] + in[k];
            }
            else {
                out[k] = in[k];
            }

            return;
        }

        __global__ void kernShift(int n, int *in, int *out) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            if (k == 0) {
                out[k] = 0;
            }
            else {
                out[k] = in[k - 1];
            }        
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            int* in;
            cudaMalloc((void**)&in, n * sizeof(int));
            int* out;
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            int roundup_n = pow(2, ilog2ceil(n));
            int blockSize = 128;
            dim3 blockPerGrid((roundup_n + blockSize - 1) / blockSize);
            
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan <<<blockPerGrid, blockSize>>>(n, pow(2, d-1), in, out);
                std::swap(in, out);
            }

            kernShift << <blockPerGrid, blockSize>> > (n, in, out);
            timer().endGpuTimer();

            cudaMemcpy(odata, out, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(in);
            cudaFree(out);
            
        }
    }
}
