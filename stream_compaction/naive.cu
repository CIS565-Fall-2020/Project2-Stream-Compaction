#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        // This version can handle arrays only as large as can be processed by a single thread block running on one multiprocessor of a GPU.
        __global__ void scan(float* g_odata, float* g_idata, int n) {
            extern __shared__ float temp[]; 
            // allocated on invocation    
            int thid = threadIdx.x;   int pout = 0, pin = 1;   
            // Load input into shared memory.   
            // This is exclusive scan, so shift right by one    
            // and set first element to 0
            temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
            __syncthreads(); 
            for (int offset = 1; offset < n; offset *= 2)   
            {     
                pout = 1 - pout; 
                // swap double buffer indices     
                pin = 1 - pout;     
                if (thid >= offset)       
                    temp[pout*n+thid] += temp[pin*n+thid - offset];     
                else       
                    temp[pout*n+thid] = temp[pin*n+thid];     
                __syncthreads();
            }   
            g_odata[thid] = temp[pout*n+thid]; 
            // write output 
        } 

        __global__ void kernScan(int n, int* odata, int*idata, int offset, int pingpong) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            // for most gpus there 1024 is the maximum number of threads per block
            int threadsPerBlock = 1024;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceiling of n / threadsPerBlock
            dim3 blockDim(threadsPerBlock, 0, 0);
            dim3 gridDim(blocksPerGrid, 0, 0);

            timer().startGpuTimer();
            int depth = ilog2ceil(n);
            for (int d = 0; d < depth; d++)
                kernScan<<<gridDim, blockDim>>>(n, odata, idata);
            timer().endGpuTimer();

            cudaFree(dev_idata);
        }
    }
}
