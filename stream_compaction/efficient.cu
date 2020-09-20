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
		
		// n: number of blocks that need to be swept
		// scaleIndex: 2^(d + 1)
		// offsetLeft: 2^(d) - 1
		// offsetRight: 2^(d + 1) - 1
		__global__ void kernUpSweep(int* oData, int nSwept, int scaleIndex, int offsetLeft, int offsetRight)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= nSwept)
			{
				return;
			}
			int k = index * scaleIndex;
			oData[k + offsetRight] += oData[k + offsetLeft];
		}

		// n: number of blocks that need to be swept
		// scaleIndex: 2^(d + 1)
		// offsetLeft: 2^(d) - 1
		// offsetRight: 2^(d + 1) - 1
		__global__ void kernDownSweep(int* oData, int nSwept, int scaleIndex, int offsetLeft, int offsetRight)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= nSwept)
			{
				return;
			}
			int k = index * scaleIndex;
			int t = oData[k + offsetLeft];
			oData[k + offsetLeft] = oData[k + offsetRight];
			oData[k + offsetRight] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int *dev_odata;
			int level = ilog2ceil(n);
			int nPOT = 1 << level;	// Clamp n to power-of-two
			cudaMalloc((void**)&dev_odata, nPOT * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata1 failed!");
			cudaMemset(dev_odata, 0, nPOT * sizeof(int));
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			

            timer().startGpuTimer();
            // DONE
			
			// Up Sweep
			int nSwept = nPOT;
			for (int d = 0; d < level; ++d)
			{
				nSwept /= 2;
				dim3 blocksPerGrid((nSwept + threadsPerBlock - 1) / threadsPerBlock);
				int scaleIndex = 1 << (d + 1);
				int offsetLeft = (1 << d) - 1;
				int offsetRight = (1 << (d + 1)) - 1;
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (dev_odata, nSwept, scaleIndex, offsetLeft, offsetRight);
			}
			// Set root to zero
			cudaMemset(dev_odata + nPOT - 1, 0, sizeof(int));
			// Down Sweep
			nSwept = 1;
			for (int d = level - 1; d >= 0; --d)
			{
				dim3 blocksPerGrid((nSwept + threadsPerBlock - 1) / threadsPerBlock);
				int scaleIndex = 1 << (d + 1);
				int offsetLeft = (1 << d) - 1;
				int offsetRight = (1 << (d + 1)) - 1;
				kernDownSweep << < blocksPerGrid, threadsPerBlock >> > (dev_odata, nSwept, scaleIndex, offsetLeft, offsetRight);
				nSwept *= 2;
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
