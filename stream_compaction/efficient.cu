#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_scan_odata;
        int* dev_dummy;
        int* dev_zeroOnes;

        __global__ void kernUpSweep(int n, int powDplus1, int powD, int* out) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n || (index % powDplus1 != 0)) {
                return;
            }
            out[index + powDplus1 - 1] += out[index + powD - 1];
        }

        __global__ void kernDownSweep(int n, int powDplus1, int powD, int* out) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n || (index % powDplus1 != 0)) {
                return;
            }
            int t = out[index + powD - 1];
            out[index + powD - 1] = out[index + powDplus1 - 1];
            out[index + powDplus1 - 1] += t;
        }



        __global__ void kernSweep(int n, int lgn, int* out) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            for (int d = 0; d <= lgn - 1; d++) {
                int powD = 1;
                for (int i = 0; i < d; i++) {
                    powD *= 2;
                }
                int powDplus1 = powD * 2;
                if (index % powDplus1 == 0) {
                    out[index + powDplus1 - 1] += out[index + powD - 1];
                }
                __syncthreads();
            }

            if (index == n - 1) {
                out[index] = 0;
            }
            __syncthreads();

            for (int d = lgn - 1; d >= 0; d--) {
                int powD = 1;
                for (int i = 0; i < d; i++) {
                    powD *= 2;
                }
                int powDplus1 = powD * 2;
                if (index % powDplus1 == 0) {
                    int t = out[index + powD - 1];
                    out[index + powD - 1] = out[index + powDplus1 - 1];
                    out[index + powDplus1 - 1] += t;
                }
                __syncthreads();
            }
        }



        __global__ void kernSetZero(int n, int* out) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            out[index] = 0;
        }

        __global__ void kernCopy(int n, int* out, int* in) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            out[index] = in[index];
        }

        


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) { // need to pad zeros if n is not pow of 2
            int lgn = ilog2ceil(n);
            int pow2Ceil = 1;
            for (int i = 0; i < lgn; i++) {
                pow2Ceil *= 2;
            }
            // set dimension
            int fullBlocksPerGrid = (pow2Ceil + blockSize - 1) / blockSize; // test 1d grid

            // cudaMalloc 
            cudaMalloc((void**)&dev_scan_odata, pow2Ceil * sizeof(int));
            cudaMalloc((void**)&dev_dummy, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");
            // cudaMemCpy
            cudaMemcpy(dev_dummy, idata, sizeof(int) * n, cudaMemcpyHostToDevice); // copy idata
            
            // set zeros
            kernSetZero << <fullBlocksPerGrid, blockSize >> > (pow2Ceil, dev_scan_odata);
            checkCUDAError("kernSetZero failed!");
            // copy dummy over to array padded with zeros
            kernCopy << <fullBlocksPerGrid, blockSize >> > (n, dev_scan_odata, dev_dummy);
            checkCUDAError("kernCopy failed!");


            timer().startGpuTimer();
            kernSweep << <fullBlocksPerGrid, blockSize >> > (pow2Ceil, lgn, dev_scan_odata);
            checkCUDAError("kernSweep failed!");
            timer().endGpuTimer();


            kernCopy << <fullBlocksPerGrid, blockSize >> > (n, dev_dummy, dev_scan_odata);
            checkCUDAError("kernCopy failed!");

            // cudaMemcpy back 
            cudaMemcpy(odata, dev_dummy, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //cudaFree
            cudaFree(dev_scan_odata);
            cudaFree(dev_dummy);
        }



        __global__ void kernZeroOnes(int n, int* out, int* in) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (in[index] != 0) {
                out[index] = 1;
            }
            else {
                out[index] = 0;
            }
        }

        __global__ void kernSweepCompact(int n, int lgn, int* out, int* dummy, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            bool isOne = false;
            if (out[index] == 1) {
                isOne = true;
            }
            for (int d = 0; d <= lgn - 1; d++) {
                int powD = 1;
                for (int i = 0; i < d; i++) {
                    powD *= 2;
                }
                int powDplus1 = powD * 2;
                if (index % powDplus1 == 0) {
                    out[index + powDplus1 - 1] += out[index + powD - 1];
                }
                __syncthreads();
            }

            if (index == n - 1) {
                out[index] = 0;
            }
            __syncthreads();

            for (int d = lgn - 1; d >= 0; d--) {
                int powD = 1;
                for (int i = 0; i < d; i++) {
                    powD *= 2;
                }
                int powDplus1 = powD * 2;
                if (index % powDplus1 == 0) {
                    int t = out[index + powD - 1];
                    out[index + powD - 1] = out[index + powDplus1 - 1];
                    out[index + powDplus1 - 1] += t;
                }
                __syncthreads();
            }

            if (isOne) {
                dummy[out[index]] = idata[index];
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
            int lgn = ilog2ceil(n);
            int pow2Ceil = 1;
            for (int i = 0; i < lgn; i++) {
                pow2Ceil *= 2;
            }
            // set dimension
            int fullBlocksPerGrid = (pow2Ceil + blockSize - 1) / blockSize; // test 1d grid

            // cudaMalloc 
            cudaMalloc((void**)&dev_scan_odata, pow2Ceil * sizeof(int));
            cudaMalloc((void**)&dev_zeroOnes, pow2Ceil * sizeof(int));
            cudaMalloc((void**)&dev_dummy, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");
            // cudaMemCpy
            cudaMemcpy(dev_dummy, idata, sizeof(int) * n, cudaMemcpyHostToDevice); // copy idata

            // set zeros
            kernSetZero << <fullBlocksPerGrid, blockSize >> > (pow2Ceil, dev_scan_odata);
            checkCUDAError("kernSetZero failed!");
            // copy dummy over to array padded with zeros
            kernCopy << <fullBlocksPerGrid, blockSize >> > (n, dev_scan_odata, dev_dummy);
            checkCUDAError("kernCopy failed!");

            // set dummy to zero for later
            kernSetZero << <fullBlocksPerGrid, blockSize >> > (n, dev_dummy);
            checkCUDAError("kernSetZero failed!");


            // --------------------------------------------------------------------------------------
            timer().startGpuTimer();
            // create ZeroOne array
            kernZeroOnes << <fullBlocksPerGrid, blockSize >> > (pow2Ceil, dev_zeroOnes, dev_scan_odata);
            checkCUDAError("kernZeroOnes failed!");

            kernSweepCompact << <fullBlocksPerGrid, blockSize >> > (pow2Ceil, lgn, dev_zeroOnes, dev_dummy, dev_scan_odata);
            checkCUDAError("kernSweepCompact failed!");

            timer().endGpuTimer();
            // --------------------------------------------------------------------------------------


            /*kernCopy << <fullBlocksPerGrid, blockSize >> > (n, dev_dummy, dev_zeroOnes);
            checkCUDAError("kernCopy failed!");*/

            int* counter = new int[pow2Ceil];
            cudaMemcpy(counter, dev_zeroOnes, sizeof(int) * pow2Ceil, cudaMemcpyDeviceToHost); // should change for non 2 pows
            int count = 0;
            count = counter[pow2Ceil - 1];

            // cudaMemcpy back 
            //cudaMemcpy(odata, dev_dummy, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_dummy, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //cudaFree
            cudaFree(dev_scan_odata);
            cudaFree(dev_dummy);
            cudaFree(dev_zeroOnes);
            delete(counter);

            return count;
        }
    }
}
