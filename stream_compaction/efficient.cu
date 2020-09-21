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
			int interval = 1 << (d + 1);
			int halfInterval = 1 << d;
			if ((index + 1) % interval == 0) {
				odata[index] += odata[index - halfInterval];
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
			int interval = 1 << (d + 1);
			int halfInterval = 1 << d;
			if ((index + 1) % interval == 0) {
				int tmp = odata[index - halfInterval];
				odata[index - halfInterval] = odata[index];
				odata[index] += tmp;
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
			int N = pow(2, ilog2ceil(n));
			int *dev_odata;
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
			cudaFree(dev_odata);
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
            // TODO
			int N = pow(2, ilog2ceil(n));
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int topLayer = ilog2ceil(n) - 1;
			int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_bools, N * sizeof(int));
			cudaMalloc((void**)&dev_indices, N * sizeof(int));
			
			timer().startGpuTimer();
			Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
			if (N > n) {
				int *zeroArray = new int[N - n];
				for (int i = 0; i < N - n; i++) {
					zeroArray[i] = 0;
				}
				cudaMemcpy(dev_bools + n, zeroArray, (N - n) * sizeof(int), cudaMemcpyHostToDevice);
			}
			cudaMemcpy(dev_indices, dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);

			int countScatter = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					countScatter++;
				}
			}
			for (int d = 0; d <= topLayer; d++) {
				kernEfficientScanUpSweep << <fullBlocksPerGrid, blockSize >> > (N, dev_indices, d);
			}

			for (int d = topLayer; d >= 0; d--) {
				kernEfficientScanDownSweep << <fullBlocksPerGrid, blockSize >> > (N, dev_indices, d, topLayer);
			}

			Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata,
				dev_idata, dev_bools, dev_indices);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, countScatter * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
            return countScatter;
        }
    }
}
