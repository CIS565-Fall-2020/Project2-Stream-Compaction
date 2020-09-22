#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <string>

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
        //__global__ void scan(float* g_odata, float* g_idata, int n) {
        //    extern __shared__ float temp[]; 
        //    // allocated on invocation    
        //    int thid = threadIdx.x;   int pout = 0, pin = 1;   
        //    // Load input into shared memory.   
        //    // This is exclusive scan, so shift right by one    
        //    // and set first element to 0
        //    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
        //    __syncthreads(); 
        //    for (int offset = 1; offset < n; offset *= 2)   
        //    {     
        //        pout = 1 - pout; 
        //        // swap double buffer indices     
        //        pin = 1 - pout;     
        //        if (thid >= offset)       
        //            temp[pout*n+thid] += temp[pin*n+thid - offset];     
        //        else       
        //            temp[pout*n+thid] = temp[pin*n+thid];     
        //        __syncthreads();
        //    }   
        //    g_odata[thid] = temp[pout*n+thid]; 
        //    // write output 
        //} 

        __global__ void kernInitExScan(int n, int* temp, int* idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= n)
                idata[idx] = 0;
            // shift the array to the right by one for exclusive scan
            // the initializing the padding of idata inn the above line is not guaranteed to be 
            // completed for all threads by the time the next line is reached
            // so just initialize all of the padding in the temp to 0 here
            temp[idx] = (idx > 0 && idx < n) ? idata[idx - 1] : 0;
        }

        __global__ void kernExScan(int pN, int* temp, int* odata, const int*idata, int offset, int pingpong) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= offset)
                temp[pingpong * pN + idx] += temp[(1 - pingpong) * pN + idx - offset];
            else
                temp[pingpong * pN + idx] = temp[(1 - pingpong) * pN + idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        using namespace std;
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            int* dev_temp;

            int depth = ilog2ceil(n);
            // remember numbers are read from right to left
            int pN = 1 << depth;    // n rounded to the next power of 2 = n after padding

            // allocating memory for dev_idata and copying memory over from idata
            cudaMalloc((void**)&dev_idata, pN * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            // std::unique_ptr<int[]>intKeys{ new int[N] };
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            // allocating memory for dev_odata
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            // allocating memory for dev_temp
            cudaMalloc((void**)&dev_temp, 2 * pN * sizeof(int));
            checkCUDAError("cudaMalloc dev_temp failed!");

            // for most gpus 1024 is the maximum number of threads per block
            int threadsPerBlock = 1024;
            int blocksPerGrid = (pN + threadsPerBlock - 1) / threadsPerBlock; // ceiling of ( pN / threadsPerBlock )
            dim3 blockDim(threadsPerBlock);
            dim3 gridDim(blocksPerGrid);

            timer().startGpuTimer();
            // initializes buffers necessary for naive exclusive scan
            kernInitExScan<<<gridDim, blockDim>>>(n, dev_temp, dev_idata);
            checkCUDAError("kernInitExScan failed!");
            // execution of naive exclusive scan in parallel
            // uses global memory instead of shared memory for ping pong buffers
            // so that the data can be of arbitrary size
            int pingpong = 0;
            for (int offset = 1; offset < pN; offset *= 2) {
                kernExScan<<<gridDim, blockDim>>>(pN, dev_temp, dev_odata, dev_idata, offset, pingpong);
                checkCUDAError("kernExScan failed!");
                pingpong = 1 - pingpong;
            }
            timer().endGpuTimer();

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_temp);
        }
    }
}
