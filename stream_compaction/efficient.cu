#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#include <cassert> 
//#include "cis565_stream_compaction_test/testing_helpers.hpp"


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpdateArray(int idx, int val, int* d_data) {
            d_data[idx] = val;
        }

        __global__ void kernUpSweepStep(
            int N,
            int d_2,
            int* d_data
        ){
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= N) {
                return;
            }
            if (k % (2 * d_2) == 0) {
                d_data[k + 2 * d_2 - 1] += d_data[k + d_2 - 1];
            }
        }

        __global__ void kernDownSweepStep(
            int N,
            int d_2,
            int* d_data
        ) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= N) {
                return;
            }

            if (k % ( d_2 * 2 )== 0) {
                int tmp = d_data[k + d_2 -1];
                d_data[k + d_2 - 1] = d_data[k + 2 * d_2 - 1];
                d_data[k + 2 * d_2 - 1] = tmp + d_data[k + 2 * d_2 - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool ifTimer = true) {
            if (n == 0) {
                return;
            }
            assert(odata != nullptr);
            assert(idata != nullptr);

            int log_n = ilog2ceil(n);
            int n_2 = 1 << log_n;

            int* dev_idata;
            dim3 blocksPerGrid = (n_2 + efficient_blocksize - 1) / efficient_blocksize;
            /*int* dev_odata;*/
            cudaMalloc((void**)&dev_idata, n_2 * sizeof(int));
            /*cudaMalloc((void**)&dev_odata, n_2 * sizeof(int));*/
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            if (ifTimer) {
                timer().startGpuTimer();
            }
            
            // TODO
            for (int d = 0; d <= log_n - 1; d ++) {
                kernUpSweepStep<<<blocksPerGrid, efficient_blocksize >>>(n_2, 1 << d, dev_idata);
            }

            //cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            kernUpdateArray<<<1, 1>>>(n_2 - 1, 0, dev_idata);
            //cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            

            for (int d = log_n - 1; d >= 0; d--) {
                kernDownSweepStep << <blocksPerGrid, efficient_blocksize >> > (n_2, 1 << d, dev_idata);
            }

            if (ifTimer) {
                timer().endGpuTimer();
            }
            
            cudaMemcpy(odata, dev_idata,n * sizeof(int), cudaMemcpyDeviceToHost);
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
        int compact(int N, int *odata, const int *idata) {
            if (N == 0) {
                return 0;
            }
            assert(odata != nullptr);
            assert(idata != nullptr);
            
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            cudaMalloc((void**)&dev_bools, N * sizeof(int));
            cudaMalloc((void**)&dev_indices, N * sizeof(int));

            cudaMemcpy(dev_idata, idata, N * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            dim3 blocksPerGrid = (N + efficient_blocksize - 1) / efficient_blocksize;
            Common::kernMapToBoolean << <blocksPerGrid, efficient_blocksize >> > (N, dev_bools, dev_idata);
            
            scan(N, dev_indices, dev_bools, false);

            Common::kernScatter << <blocksPerGrid, efficient_blocksize >> > (
                N, 
                dev_odata, 
                dev_idata,
                dev_bools,
                dev_indices
            );


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, N * sizeof(int), cudaMemcpyDeviceToHost);
            int* hst_bools = new int[N];
            cudaMemcpy(hst_bools, dev_bools, N * sizeof(int), cudaMemcpyDeviceToHost);
            int out = 0;
            for (int i = 0; i < N; i++) {
                if (hst_bools[i] == 1) {
                    out++;
                }
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return out;
        }
    }

    __global__ void prescan(float* g_odata, float* g_idata, int n) {
        extern __shared__ float temp[];  // allocated on invocation 
        int thid = threadIdx.x; int offset = 1; 
        temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory 
        temp[2*thid+1] = g_idata[2*thid+1]; 

        for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree 
        { 
            __syncthreads(); 
            if (thid < d)    
            { 
                int ai = offset * (2 * thid + 1) - 1; 
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
            }    
            offset *= 2;
        }


        if (thid == 0) { temp[n - 1] = 0; } // clear the last element  

        for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
        {      
            offset >>= 1;      
            __syncthreads();      
            if (thid < d){ 
                int ai = offset * (2 * thid + 1) - 1;     
                int bi = offset * (2 * thid + 2) - 1;

                float t = temp[ai]; 
                temp[ai] = temp[bi]; 
                temp[bi] += t;       
            } 
        }  
        __syncthreads();

        g_odata[2 * thid] = temp[2 * thid]; // write results to device memory      g_odata[2*thid+1] = temp[2*thid+1]; 

    }
}
