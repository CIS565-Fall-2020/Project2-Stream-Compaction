#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_idata;
        int* dev_odata;

        __global__ void kernUpdateK(int n, int offset, int* out, int* in) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index < offset) {
                out[index] = in[index];
            }
            else {
                out[index] = in[index - offset] + in[index];
            }
        }

        __global__ void kernShiftRight(int n, int* out, int* in) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index == 0) {
                out[index] = 0;
            }
            else {
                out[index] = in[index - 1];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // cudaMalloc 
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            // cudaMemCpy
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // set dimension
            int fullBlocksPerGrid = (n + blockSize - 1) / blockSize; // test 1d grid
            int lgn = ilog2ceil(n);
            int* temp;
            for (int d = 1; d <= lgn; d++) {
                // calculate offset from d
                int offset = 1;
                if (d > 1) {
                    for (int i = 0; i < d - 1; i++) {
                        offset *= 2;
                    }
                }
           
                kernUpdateK << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_odata, dev_idata);
                checkCUDAError("kernUpdateK failed!");
                // now o = curr, i = old.

                // ping-pong
                temp = dev_odata;
                dev_odata = dev_idata;
                dev_idata = temp;
                // now, o = old, i = curr
            }

            // shift right to make exclusive
            kernShiftRight << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata);
            checkCUDAError("kernShiftRight failed!");
            // now o = curr, i = old
            timer().endGpuTimer();

            // cudaMemcpy back 
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //cudaFree
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
