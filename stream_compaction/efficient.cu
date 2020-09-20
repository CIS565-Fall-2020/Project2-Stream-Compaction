#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
            int step = 1 << (d + 1);
            int start = step - 1;
            int index = thread * step + start;

            if (index >= n) {
                return;
            }
            data[index] += data[index - step / 2];
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
            int power1 = 1 << (d + 1);
            int power2 = power1 >> 1;

            thread = thread * power1;
            if (thread >= n) {
                return;
            }

            int right = thread + power1  - 1;
            if (right >= n) {
                return;
            }
            int left = thread + power2 - 1;
            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        __global__ void kernExtendArr(int extendNum, int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= extendNum) {
                return;
            }
            if (index >= n) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernSetValue(int n, int value, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == n) {
                data[index] = value;
            }
            else {
                return;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_extend;
            int* dev_arr;

            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Expand non power-2 to power-2
            int ceil = ilog2ceil(n);
            int num = 1 << ceil;
            int* extendData = new int[num];
            int* tmp = new int[num];

            cudaMalloc((void**)&dev_extend, num * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMalloc((void**)&dev_arr, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMemcpy(dev_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            kernExtendArr<<<fullBlocksPerGrid, threadsPerBlock>>>(num, n, dev_arr, dev_extend);

            for (int d = 0; d <= ceil; d++) {
                kernUpSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_extend);
                /*
                cudaMemcpy(tmp, dev_extend, num * sizeof(int), cudaMemcpyDeviceToHost);
                printf("_________________level %d___________________\n", d);
                for (int i = 0; i < num; i++) {
                    printf("%3d  ", tmp[i]);
                }
                printf("\n");
                */
            }
            timer().endGpuTimer();

            kernSetValue << <fullBlocksPerGrid, threadsPerBlock >> > (num - 1, 0, dev_extend);
            /*
            cudaMemcpy(tmp, dev_extend, num * sizeof(int), cudaMemcpyDeviceToHost);
            printf("SetValue\n");
            for (int i = 0; i < num; i++) {
                printf("%3d  ", tmp[i]);
            }
            printf("\n");
            */

            for (int d = ceil - 1; d >= 0; d--) {
                kernDownSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_extend);
                /*
                cudaMemcpy(tmp, dev_extend, num * sizeof(int), cudaMemcpyDeviceToHost);
                printf("_________________level %d___________________\n", d);
                for (int i = 0; i < num; i++) {
                    printf("%3d  ", tmp[i]);
                }
                printf("\n");
                */
            }

            cudaMemcpy(odata, dev_extend, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            /*
            printf("_________________test____________________\n");
            for (int i = 0; i < n; i++) {
                printf("%3d  ", odata[i]);
            }
            */
            cudaFree(dev_extend);
            cudaFree(dev_arr);
        }

        __global__ void kernMap(int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = idata[index] == 0 ? 0 : 1;
        }

        __global__ void kernScatter(int n, int* mapdata, int* scandata, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (mapdata[index] != 0) {
                odata[scandata[index]] = idata[index];
            }
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
            int* dev_map;
            int* dev_scan;
            int* dev_scatter;
            int* dev_data;
            int* host_map = new int[n];

            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_map, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");


            cudaMalloc((void**)&dev_scan, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");


            cudaMalloc((void**)&dev_scatter, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO

            // map
            kernMap << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_data, dev_map);

            // scan

            // scatter
            timer().endGpuTimer();
            return -1;
        }
    }
}
