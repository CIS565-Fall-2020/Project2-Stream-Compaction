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

#pragma region vanilla
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
#pragma endregion

#pragma region indexScale
// for part 5
        __global__ void kernUpSweepIndexScaleStep(
            int N,
            int d_2,
            int* d_data
        ) {
            // use size_t in case overflow
            size_t k = 2 * d_2 * ( (blockIdx.x * blockDim.x) + threadIdx.x) + 2 * d_2 - 1;
            /*k *= 2 * d_2;
            k += 2 * d_2 - 1;*/
            if (k >= N) {
                return;
            }
            d_data[k] += d_data[k - d_2];
        }

        __global__ void kernDownSweepIndexScaleStep(
            int N,
            int d_2,
            int* d_data
        ) {
            size_t k = 2 * d_2 * ((blockIdx.x * blockDim.x) + threadIdx.x) + 2 * d_2 - 1;
            if (k >= N) {
                return;
            }
            int tmp = d_data[k 
                - d_2];
            d_data[k - d_2] = d_data[k];
            d_data[k] = tmp + d_data[k];
        }
#pragma endregion

#pragma region SharedMemory
        __global__ void kernSharedMemoryUpSweepStep(int N, int d_2, int cur_depth, int target_depth, int* dev_idata) {
            size_t t_offset = blockIdx.x * blockDim.x;
            size_t t_id = threadIdx.x;
            size_t k = 2 * d_2 * (t_offset + t_id) + 2 * d_2 - 1;
            if (k >= N) {
                return;
            }

            extern __shared__ float shared[];
            shared[2 * t_id] = dev_idata[k - d_2];
            shared[2 * t_id + 1] = dev_idata[k];
            __syncthreads();

            /*shared[2 * t_id + 1] += shared[2 * t_id];*/
            
            for (int i = 0; i < target_depth - cur_depth; i++) {
                int mul = 1 << (i + 1);
                int idx_a = mul * (t_id + 1) - 1;
                int idx_b = mul * (t_id + 1) - mul / 2 - 1;
                if (idx_a < 2 * blockDim.x) {
                    /*int a = shared[idx_a];
                    int b = shared[idx_b];*/
                    shared[idx_a] += shared[idx_b];
                }
                __syncthreads();
            }
            

            dev_idata[k] = shared[2 * t_id + 1];
            dev_idata[k - d_2] = shared[2 * t_id];
        }

        __global__ void kernSharedMemoryDownSweepStep(int N, int d_2, int cur_depth, int target_depth, int* dev_idata) {
            size_t t_offset = blockIdx.x * blockDim.x;
            size_t t_id = threadIdx.x;
            size_t k = 2 * d_2 * (t_offset + t_id) + 2 * d_2 - 1;
            if (k >= N) {
                return;
            }

            extern __shared__ float shared[];
            shared[2 * t_id] = dev_idata[k - d_2];
            shared[2 * t_id + 1] = dev_idata[k];
            __syncthreads();

            for (int i = cur_depth - 1 - target_depth; i >= 0; i--) {
                int mul = 1 << (i + 1);
                int idx_a = mul * (t_id + 1) - 1;
                int idx_b = mul * (t_id + 1) - mul / 2 - 1;
                if (idx_a < 2 * blockDim.x) {
                    /*int a = shared[idx_a];
                    int b = shared[idx_b];*/
                    int tmp = shared[idx_b];
                    shared[idx_b] = shared[idx_a];
                    shared[idx_a] += tmp;
                }
                __syncthreads();
            }

            dev_idata[k - d_2] = shared[2 * t_id];
            dev_idata[k] = shared[2 * t_id + 1];
        }
 #pragma endregion
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool ifTimer = true,bool ifIdxScale = false, bool ifSharedMemory = false) {
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
            if (ifSharedMemory) {
                int unroll_depth = ilog2ceil(efficient_blocksize);
                
                for (int cur_depth = 0; cur_depth < log_n; cur_depth += unroll_depth) {
                    int d = cur_depth;
                    blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                    int target_depth = std::min(cur_depth + unroll_depth, log_n); // log_n exclusive
                    kernSharedMemoryUpSweepStep << <blocksPerGrid, efficient_blocksize, 2 * efficient_blocksize * sizeof(int) >> > (n_2, 1 << d, cur_depth, target_depth, dev_idata);
                }
            }
            else {
                for (int d = 0; d <= log_n - 1; d++) {
                    if (ifIdxScale) {
                        blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                        kernUpSweepIndexScaleStep << <blocksPerGrid, efficient_blocksize >> > (n_2, 1 << d, dev_idata);
                    }
                    /*else if (ifSharedMemory) {
                        blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                        kernSharedMemoryUpSweepStep <<<blocksPerGrid, efficient_blocksize, 2 * efficient_blocksize * sizeof(int) >>> (n_2, 1 << d, dev_idata);
                    }*/
                    else {
                        kernUpSweepStep << <blocksPerGrid, efficient_blocksize >> > (n_2, 1 << d, dev_idata);
                    }
                }
            }
            

            kernUpdateArray << <1, 1 >> > (n_2 - 1, 0, dev_idata);

            if (ifSharedMemory) {
                int unroll_depth = ilog2ceil(efficient_blocksize);
                for (int cur_depth = log_n; cur_depth > 0; cur_depth -= unroll_depth) {
                    int target_depth = std::max(0, cur_depth - unroll_depth);
                    int d = target_depth;
                    blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                    kernSharedMemoryDownSweepStep << <blocksPerGrid, efficient_blocksize, 2 * efficient_blocksize * sizeof(int) >> > (n_2, 1 << d, cur_depth, target_depth, dev_idata);
                }
            }
            else {
                for (int d = log_n - 1; d >= 0; d--) {
                    if (ifIdxScale) {
                        blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                        kernDownSweepIndexScaleStep << <blocksPerGrid, efficient_blocksize >> > (n_2, 1 << d, dev_idata);
                    }
                    /*else if (ifSharedMemory) {
                        blocksPerGrid = (n_2 / (1 << (1 + d)) + efficient_blocksize - 1) / efficient_blocksize;
                        kernSharedMemoryDownSweepStep << <blocksPerGrid, efficient_blocksize, 2 * efficient_blocksize * sizeof(int) >> > (n_2, 1 << d, dev_idata);
                    }*/
                    else {
                        kernDownSweepStep << <blocksPerGrid, efficient_blocksize >> > (n_2, 1 << d, dev_idata);
                    }
                }
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
            
            scan(N, dev_indices, dev_bools, false, true, false);

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



    // ref ::  gpu gem
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
