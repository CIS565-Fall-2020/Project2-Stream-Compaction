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

        // DONE: __global__
		__global__ void kernNaiveScan(int *odata, const int *idata, int offset, int n)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n)
			{
				return;
			}
			// offset = 2^(d-1)
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
        void scan(int n, int *odata, const int *idata) {
			int *dev_idata;
			int *dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata0 failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata1 failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // DONE
			dim3 blocksPerGrid((n + threadsPerBlock - 1) / threadsPerBlock);
			int dmax = ilog2ceil(n);
			for (int d = 1; d <= dmax; ++d)
			{
				int offset = 1 << (d - 1);
				kernNaiveScan << <blocksPerGrid, threadsPerBlock >> > (dev_odata, dev_idata, offset, n);
				std::swap(dev_odata, dev_idata);
			}
			
            timer().endGpuTimer();
			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
