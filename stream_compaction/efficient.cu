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

        __global__ void kernEfficientUpSweep(int n, int offset, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            index = (index + 1) * offset * 2 - 1;
            if (index >= n) {
                return;
            }
            data[index] += data[index - offset];
        }

        __global__ void kernEfficientDownSweep(int n, int offset, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            index = (index + 1) * 2 * offset - 1;
            if (index >= n) {
                return;
            }
            int temp = data[index - offset];
            data[index - offset] = data[index];
            data[index] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) { // TODO
            if (n < 1) {
                return;
            }
            int d = ilog2ceil(n);
            int full = 1 << d;
            int* dev_data = nullptr;

            // Allocate memory on device
            cudaMalloc((void**)&dev_data, full * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            // Set additional memory values to zero
            cudaMemset(dev_data + n, 0, (full - n) * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            // Copy data from host to device
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            timer().startGpuTimer();

            dim3 threadsPerBlock(128);

            // Up-Sweep
            int offset = 1;
            for (int i = 0; i < d; i++) {
                int nodeNumber = 1 << (d - i - 1);
                dim3 blocks(nodeNumber / threadsPerBlock.x + 1);
                kernEfficientUpSweep << <blocks, threadsPerBlock >> >
                    (full, offset, dev_data);
                offset <<= 1;
            }

            // Down-Sweep
            cudaMemset(dev_data + full - 1, 0, sizeof(int));
            for (int i = 0; i < d; i++) {
                offset >>= 1;
                int nodeNumber = 1 << i;
                dim3 blocks(nodeNumber / threadsPerBlock.x + 1);
                kernEfficientDownSweep << <blocks, threadsPerBlock >> >
                    (full, offset, dev_data);
            }
            

            timer().endGpuTimer();

            // Copy data back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back failed!");

            // Free device memory
            cudaFree(dev_data);
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
