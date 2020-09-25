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

        __global__ void kernaldownsweep(int n, int d, int* input) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            // 2 ^ d
            int pow_2d = 1 << d;
            // 2 ^ (d+1)
            int pow_2d1 = 1 << (d + 1);

            // we want the even indices to add into the odd indices
            if (k % pow_2d1 == 0) {
                int t = input[k + pow_2d - 1];
                input[k + pow_2d - 1] = input[k + pow_2d1 - 1];
                input[k + pow_2d1 - 1] += t;
            }
        }

        __global__ void kernalupsweep(int n, int d, int* input) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            // 2 ^ d
            int pow_2d = 1 << d;
            // 2 ^ (d+1)
            int pow_2d1 = 1 << (d + 1);

            // we want the even indices to add into the odd indices
            if (k % pow_2d1 == 0) {
                input[k + pow_2d1 - 1] += input[k + pow_2d - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // The idea is to build a balanced binary tree on the input data and 
            // sweep it to and from the root to compute the prefix sum. A binary 
            // tree with n leaves has d = log2 n levels, and each level d has 2 d nodes.

            int padded_size = 1 << ilog2ceil(n);
            int* temp_array = new int[padded_size];

            // make sure to pad temp array with 0s!
            // to do: is this faster or cuda memcpying 0s faster? hmm
            for (int i = 0; i < padded_size; i++) {
                if (i < n) {
                    temp_array[i] = idata[i];
                }
                else {
                    temp_array[i] = 0;
                }
            }

            // initialize some temporary buffers to write in place
            // your intermediate array sizes will need to be rounded to the next power of two.
            int* temp_input;
            cudaMalloc((void**)&temp_input, padded_size * sizeof(int));
            // fill temp input buffer with the padded array above
            cudaMemcpy(temp_input, temp_array, padded_size * sizeof(int), cudaMemcpyHostToDevice);

            // set up the blocks and grids
            int blockSize = 64;
            dim3 blocksPerGrid((padded_size + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            timer().startGpuTimer();
            
            // The algorithm consists of two phases : 
            // the reduce phase(also known as the up - sweep phase) 
            // and the down - sweep phase.

            // up sweep phase
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernalupsweep << <blocksPerGrid, threadsPerBlock >> > (padded_size, d, temp_input);
            }

            // replace last index as 0
            int zero = 0;
            cudaMemcpy(temp_input + padded_size - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // downsweep phase
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernaldownsweep << <blocksPerGrid, threadsPerBlock >> > (padded_size, d, temp_input);
            }

            timer().endGpuTimer();

            // copy from GPU to CPU
            cudaMemcpy(temp_array, temp_input, padded_size * sizeof(int), cudaMemcpyDeviceToHost);
        
            // copy into outdata
            for (int i = 0; i < n; i++) {
                odata[i] = temp_array[i];
            }
        
            // cleanup
            cudaFree(temp_input);
            delete[] temp_array;
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
            // we want to setup stuff for scans because we don't want the setup to be within the timer
            // your intermediate array sizes will need to be rounded to the next power of two.
            int padded_size = 1 << ilog2ceil(n);
            int* temp_array = new int[padded_size];

            int* temp_bool; // stores the bool array on gpu
            cudaMalloc((void**)&temp_bool, padded_size * sizeof(int));

            int* temp_scan_output; // stores the scanned bool array
            cudaMalloc((void**)&temp_scan_output, padded_size * sizeof(int));

            // make sure to pad temp array with 0s!
            // to do: is this faster or cuda memcpying 0s faster? hmm
            for (int i = 0; i < padded_size; i++) {
                if (i < n) {
                    temp_array[i] = idata[i];
                }
                else {
                    temp_array[i] = 0;
                }
            }

            int* temp_input;  // stores the padded input on the gpu
            cudaMalloc((void**)&temp_input, padded_size * sizeof(int));
            // fill with padded data from above
            cudaMemcpy(temp_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* temp_output; // stores the output of the scatter on the gpu
            cudaMalloc((void**)&temp_output, padded_size * sizeof(int));

            // set up the blocks and grids
            int blockSize = 128;
            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((padded_size + blockSize - 1) / blockSize);

            timer().startGpuTimer(); // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            // similar to the cpu... we want to: 
            // fill temp array with 0 if idata is 0 or 1 otherwise
            // ================= MAP TO BOOL =======================
            Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlock >> > (padded_size, temp_bool, temp_input);

            // ================= SCAN ===========================
            // The algorithm consists of two phases : 
           // the reduce phase(also known as the up - sweep phase) 
           // and the down - sweep phase.

            cudaMemcpy(temp_scan_output, temp_bool, padded_size * sizeof(int), cudaMemcpyDeviceToDevice);

           // up sweep phase
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernalupsweep << <blocksPerGrid, threadsPerBlock >> > (padded_size, d, temp_scan_output);
            }

            // replace last index as 0
            int zero = 0;
            cudaMemcpy(temp_scan_output + padded_size - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // downsweep phase
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernaldownsweep << <blocksPerGrid, threadsPerBlock >> > (padded_size, d, temp_scan_output);
            }

            // ================= SCATTER =======================
            Common::kernScatter << <blocksPerGrid, threadsPerBlock >> > (padded_size, temp_output, temp_input, temp_bool, temp_scan_output);
            timer().endGpuTimer(); // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            // we want to copy information from gpu to cpu now
            cudaMemcpy(odata, temp_output, padded_size * sizeof(int), cudaMemcpyDeviceToHost);
            // we also want to print the result of the scan so we can count how many non-zeros there are
            int result = -1;
            cudaMemcpy(&result, temp_scan_output + padded_size - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // cleanup
            cudaFree(temp_output);
            cudaFree(temp_input);
            cudaFree(temp_bool);
            cudaFree(temp_scan_output);
            delete[] temp_array;

            return result;
        }
    }
}
