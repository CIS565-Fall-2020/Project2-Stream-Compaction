#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernEfficientScanUpSweep(int n, int *odata, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int interval = (int)pow(2.0, (double)(d + 1));
			if ((index + 1) % interval == 0) {
				odata[index] += odata[index - interval / 2];
			}
		}

		__global__ void kernEfficientScanDownSweep(int n, int *odata, int d, int topLayer) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (d == topLayer && index == n - 1) {
				odata[index] = 0;
			}
			int interval = (int)pow(2.0, (double)(d + 1));
			if ((index + 1) % interval == 0) {
				int tmp = odata[index - interval / 2];
				odata[index - interval / 2] = odata[index];
				odata[index] += tmp;
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
			int N = pow(2, ilog2ceil(n));
			int *dev_odata, *dev_tmp;
			cudaMalloc((void**)&dev_odata, N * sizeof(int));
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			if (N > n) {
				int *zeroArray = new int[N - n];
				for (int i = 0; i < N - n; i++) {
					zeroArray[i] = 0;
				}
				cudaMemcpy(dev_odata + n, zeroArray, (N - n) * sizeof(int), cudaMemcpyHostToDevice);
			}
			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int topLayer = ilog2ceil(n) - 1;
			for (int d = 0; d <= topLayer; d++) {
				kernEfficientScanUpSweep << <fullBlocksPerGrid, blockSize >> > (N, dev_odata, d);
			}
			
			for (int d = topLayer; d >= 0; d--) {
				kernEfficientScanDownSweep << <fullBlocksPerGrid, blockSize >> > (N, dev_odata, d, topLayer);
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
