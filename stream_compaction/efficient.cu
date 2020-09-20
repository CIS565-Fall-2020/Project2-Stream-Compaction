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

        __global__ void kernEfficientUpSweep(int n, int offset,
            int numNode, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= numNode) {
                return;
            }
            index = (index + 1) * offset * 2 - 1;
            
            data[index] += data[index - offset];
        }

        __global__ void kernEfficientDownSweep(int n, int offset,
            int numNode, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= numNode) {
                return;
            }
            index = (index + 1) * 2 * offset - 1;

            int temp = data[index - offset];
            data[index - offset] = data[index];
            data[index] += temp;
        }

        void scanHelper(int full, int d, int blockSize, int *dev_data) {
            dim3 threadsPerBlock(blockSize);

            // Up-Sweep
            int offset = 1;
            for (int i = 0; i < d; i++) {
                int numNode = 1 << (d - i - 1);
                dim3 blocks(numNode / threadsPerBlock.x + 1);
                kernEfficientUpSweep << <blocks, threadsPerBlock >> >
                    (full, offset, numNode, dev_data);
                offset <<= 1;
            }

            // Down-Sweep
            cudaMemset(dev_data + full - 1, 0, sizeof(int));
            for (int i = 0; i < d; i++) {
                offset >>= 1;
                int numNode = 1 << i;
                dim3 blocks(numNode / threadsPerBlock.x + 1);
                kernEfficientDownSweep << <blocks, threadsPerBlock >> >
                    (full, offset, numNode, dev_data);
            }
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

            int blockSize = 32;
            scanHelper(full, d, blockSize, dev_data);            

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
        int compact(int n, int *odata, const int *idata) { // TODO
            if (n < 1) {
                return 0;
            }
            int* dev_data = nullptr;
            int* dev_bools = nullptr;
            int* dev_indices = nullptr;
            int* dev_final = nullptr;
            
            int d = ilog2ceil(n);
            int full = 1 << d;

            // Allocate memory on device
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, full * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_final, full * sizeof(int));
            checkCUDAError("cudaMalloc dev_final failed!");

            // Copy data to device
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            // Set additional elements zero
            cudaMemset(dev_indices + n, 0, (full - n) * sizeof(int));
            checkCUDAError("cudaMemset dev_bools failed!");

            timer().startGpuTimer();

            int blockSize = 128;

            dim3 threadsPerBlock(blockSize);
            dim3 blocks(n / threadsPerBlock.x + 1);

            StreamCompaction::Common::kernMapToBoolean
                << <blocks, threadsPerBlock >> > (n, dev_bools, dev_data);

            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int),
                cudaMemcpyDeviceToDevice);
            
            scanHelper(full, d, blockSize, dev_indices);
            
            StreamCompaction::Common::kernScatter<< <blocks, threadsPerBlock >> >
                (n, dev_final, dev_data, dev_bools, dev_indices);

            timer().endGpuTimer();

            // Get the number of remaining elements
            int lastIndex;
            int lastBool;
            cudaMemcpy((void*)&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int remain = lastIndex + lastBool;

            // Copy data back to host
            cudaMemcpy(odata, dev_final, remain * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_data);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_final);
            
            return remain;
        }
    }
}
