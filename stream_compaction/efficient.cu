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


        __global__ void prescan(float* g_odata, float* g_idata, int n) {
            extern __shared__ float temp[];
            // allocated on invocation 
            int thid = threadIdx.x; int offset = 1;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            // for most gpus there are 1024 threads per block
            int threadsPerBlock = 1024;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceiling of n / threadsPerBlock
            dim3 blockDim(threadsPerBlock, 0, 0);
            dim3 gridDim(blocksPerGrid, 0, 0);


            timer().startGpuTimer();
            // TODO
            int k = ilog2ceil(n);
            kernScan << <gridDim, blockDim >> > ();

            timer().endGpuTimer();

            cudaFree(dev_idata);
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
