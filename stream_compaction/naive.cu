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
        
        __global__ void kernalNaiveScan(int n, int d, int* input, int* output) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            // if k >= 2 ^ (d - 1) <-- see example 2 in Ch 39 Patch
            if (k >= (1 << (d - 1))) {
                output[k] = input[k - (1 << (d - 1))] + input[k];
            }
            else {
                output[k] = input[k];
            }
        }

        __global__ void kernalInc2Exc(int n, int* input, int* output) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            // shift everything to the right
            // the default is 0
            if (k == 0) {
                output[k] = 0;
            }
            else {
                output[k] = input[k - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // set up the blocks and grids
            int blockSize = 64; 
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            // initialize some temporary buffers to write in place
            int* temp_input;
            cudaMalloc((void**)&temp_input, n * sizeof(int));
            // fill temp input buffer with the original input
            cudaMemcpy(temp_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* temp_output;
            cudaMalloc((void**)&temp_output, n * sizeof(int));

            timer().startGpuTimer();
            // iterate through for d = 1 to d = ilog2ceil(n)
            for (int d = 1; d <= ilog2ceil(n); d++) {
                // during each time, we want to call kernel to parallel scan
                // from input to output
                kernalNaiveScan<<<blocksPerGrid, threadsPerBlock>>>(n, d, temp_input, temp_output);

                // remember to swap the buffers!
                std::swap(temp_input, temp_output);
            }

            // we want an exclusive scan so we have to convert
            kernalInc2Exc << <blocksPerGrid, threadsPerBlock >> > (n, temp_input, temp_output);
           
            timer().endGpuTimer();

            // now we want to write everything to our real output buffer
            cudaMemcpy(odata, temp_output, n * sizeof(int), cudaMemcpyDeviceToHost);

            // cleanup
            cudaFree(temp_input);
            cudaFree(temp_output);
        }
    }
}
