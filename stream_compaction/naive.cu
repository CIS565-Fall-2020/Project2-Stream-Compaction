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
        // TODO: __global__
        // This performs inclusive scan
        __global__ void kernNaiveScan(int d, int n, int *odata, int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            int d_offset = 1 << (d - 1);

            if(index >= d_offset) 
            {
                odata[index] = idata[index - d_offset] + idata[index];
            }
            else 
            {   
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int *dev_idata, *dev_odata;
            

            //dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            // CUDA memory management and error checking
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_idata failed!");

            timer().startGpuTimer();

            for(int d = 1; d <= ilog2ceil(n); ++d)
            {   
                kernNaiveScan<<<blocksPerGrid, blockSize>>>(d, n, dev_odata, dev_idata);

                int *dev_temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = dev_temp;
            }

            timer().endGpuTimer();
            
            // Right shift copy to achieve exclusive scan
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_idata to odata + 1 failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
