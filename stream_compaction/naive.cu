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
        // TODO: __global__
        __global__ void kernScan(int n, int bar, int *in, int *out) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;
            if (k >= bar) {
                out[k] = in[k - bar] + in[k];
            }

            return;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* in;
            cudaMalloc((void**)&in, n * sizeof(int));
            int* out;
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan <<<1, n>>>(n, pow(2, d-1), in, out);
                std::swap(in, out);
            }
            
            cudaMemcpy(odata, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
            //std::cout << in[0] << std::endl;
            //for (int i = 0; i < n; i++) {
                //odata[i] = in[i];
            //    std::cout << in[i] << std::endl;
            //}
            
            cudaFree(in);
            cudaFree(out);
            
            timer().endGpuTimer();
        }
    }
}
