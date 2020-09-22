#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <stdio.h>
#define GLM_FORCE_CUDA
#define blockSize 1024

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_A;
        int* dev_B; 
       
        // TODO: 
        //This kernal performs an inclusive scan on the in array and stores it in the out array 
        __global__ void kernNaiveParallelScanInclusive(int offset, int n, int* in, int* out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;

            if (index >= offset)
            {
                out[index] = in[index - offset] + in[index];
            }
            else
            {
                out[index] = in[index]; 
            }
        }

        //This kernal shifts the elements to right and sets the first elem to 0
        //Inclusive to exclusive 
        __global__ void kernConvertToExclusive(int n, int* in, int* out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;
            if (index == 0)
            {
                out[0] = 0;
            }
            else
            {
                out[index] = in[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            //Allocate memory on device 
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc A failed!");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc B failed!");

            //Copy idata from host into device A and device B 
            cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy host to device A failed!");
            cudaMemcpy(dev_B, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy host to device B failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            //Loop similar to the implementation in 
            //http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            //Look at Page 10 
            for (int d = 1; d < n; d <<= 1)
            {
                kernNaiveParallelScanInclusive << <fullBlocksPerGrid, blockSize >> > (d, n, dev_A, dev_B);
                checkCUDAErrorFn("kernNaiveParallelScanInclusive failed!");
                std::swap(dev_A, dev_B);
            }
            kernConvertToExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_A, dev_B);
            checkCUDAErrorFn("kernConvertToExclusive failed!");

            timer().endGpuTimer();

            //Copy back device B into host odata 
            cudaMemcpy(odata, dev_B, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy device B to host failed!");

            //for (int i = 0; i < 10; ++i)
            //{
            //    printf(" odata is %d \n", odata[i]);
            //}

            //Free the device memory 
            cudaFree(dev_A);
            cudaFree(dev_B); 

        }

        //This version can handle arrays only as large as can be processed by a single thread block running on 
        //one multiprocessor of a GPU.
        //__global__ void scanSharedMem(float* g_odata, float* g_idata, int n) 
        //{
        //    extern __shared__ float temp[]; // allocated on invocation    
        //    int thid = threadIdx.x;   
        //    int pout = 0, pin = 1;   // Load input into shared memory.                                      
        //                             // This is exclusive scan, so shift right by one    
        //                             // and set first element to 0   

        //    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;   
        //    __syncthreads();   

        //    for (int offset = 1; offset < n; offset *= 2)
        //    {
        //        pout = 1 - pout; // swap double buffer indices     
        //        pin = 1 - pout;
        //        if (thid >= offset)
        //        {             
        //            temp[pout * n + thid] += temp[pin * n + thid - offset];
        //        }
        //        else
        //        {
        //            temp[pout * n + thid] = temp[pin * n + thid];
        //        }
        //        __syncthreads();   
        //    }   
        //    g_odata[thid] = temp[pout*n+thid]; // write output 
        //} 
    }
}
