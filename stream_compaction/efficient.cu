#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernScan1(int n, int pow_d, int pow_d_1, int* in) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            k = k * pow_d_1;
            if (k >= n) {
                return;
            }
            
            in[k + pow_d_1 - 1] += in[k + pow_d - 1];
            return;
        }

        __global__ void kernScan2(int n, int pow_d, int pow_d_1, int* in) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            k = k * pow_d_1;
            if (k >= n) {
                return;
            }
            
            int t = in[k + pow_d - 1];
            in[k + pow_d - 1] = in[k + pow_d_1 - 1];
            in[k + pow_d_1 - 1] += t;
            return;
        }

        __global__ void kernPadZero(int idx, int roundup, int* in) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= idx && k < roundup) {
                in[k] = 0;
            }
            return;
        }

        __global__ void kernShift(int n, int* in, int* out) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            if (k == 0) {
                out[k] = 0;
            }
            else {
                out[k] = in[k - 1];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;
            int roundup_n = pow(2, ilog2ceil(n));

            int* in;
            cudaMalloc((void**)&in, roundup_n * sizeof(int));
            int* out;
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            dim3 blockPerGrid((roundup_n + blockSize - 1) / blockSize);
            kernPadZero << <blockPerGrid, roundup_n>>>(n, roundup_n, in);
            int num;
            for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
                num = roundup_n / pow(2, d + 1);
                dim3 blockPerGridLoop1((num + blockSize - 1) / blockSize);
                kernScan1 << <blockPerGridLoop1, num >> > (roundup_n, pow(2, d), pow(2, d+1), in);
                cudaMemcpy(odata, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 20; i++) {
                    std::cout << odata[i] << " ";
                }
                std::cout << std::endl;
            }
            
            //kernPadZero << <blockPerGrid, roundup_n >> > (roundup_n - 1, roundup_n, in);
            cudaMemset(in + roundup_n - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                num = roundup_n / pow(2, d + 1);
                dim3 blockPerGridLoop2((num + blockSize - 1) / blockSize);
                kernScan2 << <blockPerGridLoop2, num >> > (roundup_n, pow(2, d), pow(2, d + 1), in);
            }
            /*
            dim3 blockPerGridShift((n + blockSize - 1) / blockSize);
            kernShift << <blockPerGridShift, blockSize >> > (n, in, out);
            */
            timer().endGpuTimer();
            cudaMemcpy(odata, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 20; i++) {
                std::cout << odata[i] << " ";
            }
            std::cout << std::endl;
            cudaFree(in);
            cudaFree(out);
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
