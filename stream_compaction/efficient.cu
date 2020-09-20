#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // up sweep
        __global__ void upSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int dist = pow(2.f, d + 1);
            if (index >= n || index % dist != 0) {
                return;
            }
            int toUpdate = index + pow(2.f, d + 1) - 1;
            int toGet = index + pow(2.f, d) - 1;
            data[toUpdate] += data[toGet];
        }

        // down sweep
        __global__ void downSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int dist = pow(2.f, d + 1);
            if (index >= n || index % dist != 0) {
                return;
            }
            int t_index = index + pow(2.f, d) - 1;
            int replace_index = index + pow(2.f, d + 1) - 1;
            int t = data[t_index];
            data[t_index] = data[replace_index];
            data[replace_index] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            //timer().startGpuTimer();

            int power_of_2 = 1;
            while (power_of_2 < n) {
                power_of_2 *= 2;
            }

            // create array of size power of 2
            int* data;

            cudaMalloc((void**)&data, power_of_2 * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc data failed!");

            // fill array and pad end with 0's
            std::unique_ptr<int[]>padded_array{ new int[power_of_2] };
            cudaMemcpy(padded_array.get(), idata, sizeof(int) * n, cudaMemcpyHostToHost);
            for (int i = n; i < power_of_2; i++) {
                padded_array[i] = 0;
            }
            cudaMemcpy(data, padded_array.get(), sizeof(int) * power_of_2, cudaMemcpyHostToDevice);

            // kernel values
            int blockSize = 128;
            dim3 fullBlocksPerGrid((power_of_2 + blockSize - 1) / blockSize);

            // up-sweep
            for (int d = 0; d <= ilog2(power_of_2) - 1; d++) {
                upSweep << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data);
            }

            // set the last value to 0
            cudaMemcpy(padded_array.get(), data, sizeof(int) * power_of_2, cudaMemcpyDeviceToHost);
            for (int i = n - 1; i < power_of_2; i++) {
                padded_array[i] = 0;
            }
            cudaMemcpy(data, padded_array.get(), sizeof(int) * power_of_2, cudaMemcpyHostToDevice);

            // down-sweep
            for (int d = ilog2(power_of_2) - 1; d >= 0; d--) {
                downSweep << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data);
            }

            // set the out data to the scanned data
            cudaMemcpy(odata, data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(data);

            //timer().endGpuTimer();
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
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();

            // malloc necessary space oon GPU
            int* zerosAndOnes;
            int* scanned_data;
            int* scattered_data;

            cudaMalloc((void**)&zerosAndOnes, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc zerosAndOnes failed!");

            cudaMalloc((void**)&scanned_data, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc scanned_data failed!");

            cudaMalloc((void**)&scattered_data, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc scattered_data failed!");

            // change to zeros and ones
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, zerosAndOnes, idata);

            // exclusive scan data
            scan(n, scanned_data, zerosAndOnes);

            // scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, scattered_data, idata, zerosAndOnes, scanned_data);
            cudaMemcpy(odata, scattered_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // return last index in scanned_data
            std::unique_ptr<int[]>scanned_cpu{ new int[n] };
            cudaMemcpy(scanned_cpu.get(), scanned_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
            return scanned_cpu[n] + 1;
        }
    }
}
