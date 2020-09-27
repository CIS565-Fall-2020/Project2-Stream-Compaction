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
        __global__ void kernScan1(int n, int d, int* in) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int pow_d_1 = 1 << (d + 1);
            int pow_d = 1 << d;
            if (k >= n / pow_d_1) {
                return;
            }
            k = k * pow_d_1;
            in[k + pow_d_1 - 1] += in[k + pow_d - 1]; // 1 += 0
            return;
        }

        __global__ void kernScan2(int n, int pow_d, int pow_d_1, int* in) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n / pow_d_1) {
                return;
            }
            k = k * pow_d_1;
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

        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;
            int roundup_n = pow(2, ilog2ceil(n));

            int* in;
            cudaMalloc((void**)&in, roundup_n * sizeof(int));
            cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            //timer().startGpuTimer();
            
            dim3 blockPerGrid((roundup_n + blockSize - 1) / blockSize);
            kernPadZero << <blockPerGrid, roundup_n>>>(n, roundup_n, in);
            int num = 0;
            for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
                num = roundup_n / pow(2, d + 1);
                dim3 blockPerGridLoop1((num + blockSize - 1) / blockSize);
                kernScan1 << <blockPerGridLoop1, blockSize >> > (roundup_n, d, in);
            }
            //kernPadZero << <blockPerGrid, roundup_n >> > (roundup_n - 1, roundup_n, in);
            cudaMemset(in + roundup_n - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                num = roundup_n / (1 << (d + 1));
                dim3 blockPerGridLoop2((num + blockSize - 1) / blockSize);
                kernScan2 << <blockPerGridLoop2, blockSize >> > (roundup_n, 1 << d, 1 << (d + 1), in);
            }
            //timer().endGpuTimer();
            cudaMemcpy(odata, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(in);
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
            int blockSize = 128;
            int roundup_n = pow(2, ilog2ceil(n));
            int* in;
            cudaMalloc((void**)&in, n * sizeof(int));
            int* out;
            cudaMalloc((void**)&out, n * sizeof(int));
            int* scan_res;
            cudaMalloc((void**)&scan_res, n * sizeof(int));
            int* bools;
            cudaMalloc((void**)&bools, n * sizeof(int));
            cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            int ctr = 0;
            timer().startGpuTimer();
            dim3 blockPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean << <blockPerGrid ,blockSize>> > (n, bools, in);
            scan(n, scan_res, bools);
            StreamCompaction::Common::kernScatter << <blockPerGrid, blockSize>> > (n, out, in, bools, scan_res);
            timer().endGpuTimer();
            int* bools_last = new int[0];
            cudaMemcpy(bools_last, bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int* scan_res_last = new int[0];
            cudaMemcpy(scan_res_last, scan_res + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (bools_last[0] == 1) {
                ctr = scan_res_last[0] + 1;
            }
            else {
                ctr = scan_res_last[0];
            }
            
            cudaMemcpy(odata, out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(in);
            cudaFree(out);
            cudaFree(scan_res);
            cudaFree(bools);
            delete(bools_last);
            delete(scan_res_last);
            return ctr;
        }
    }
}
