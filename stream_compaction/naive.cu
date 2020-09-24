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
        
        __global__ void kernParallelScan(int n, int* odata, const int* idata, int d)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            int offset = 1 << d;
            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_tempData;
            int* dev_outputData;
            cudaMalloc((void**)&dev_tempData, n * sizeof(int));
            cudaMalloc((void**)&dev_outputData, n * sizeof(int));
            cudaMemcpy(dev_outputData, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // ------------------------------------- Performance Measurement ------------------------------------------
            timer().startGpuTimer();
            int depth = ilog2ceil(n); 
            for (int d = 0; d < depth; d++)
            {
                kernParallelScan<<<blocksPerGrid, threadsPerBlock>>>(n, dev_tempData, dev_outputData, d);
                std::swap(dev_tempData, dev_outputData);
            }
            timer().endGpuTimer();
            // --------------------------------------------------------------------------------------------------------

            // Do a right shift when copying data from gpu to cpu, to convert inclusive scan to exclusive scan
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_outputData, (n - 1) * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(dev_tempData);
            cudaFree(dev_outputData);
        }
    }
}
