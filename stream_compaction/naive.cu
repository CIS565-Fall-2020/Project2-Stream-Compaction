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
		// TODO: __global__
		__global__ void kernNaiveScan(int n, int *odata, int *tmp, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			tmp[index] = odata[index];
			int offset = (int)pow(2.0, (double)(d - 1));
			if (index >= offset) {
				tmp[index] = odata[index - offset] + odata[index];
			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			// TODO
			int *dev_odata, *dev_tmp;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			int *zero = 0;
			cudaMemcpy(dev_odata, zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&dev_tmp, n * sizeof(int));

			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_tmp, d);
				int *tmpPtr = dev_tmp;
				dev_tmp = dev_odata;
				dev_odata = tmpPtr;
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_tmp);
		}
	}
}
