#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>
#include <cassert> // Jack12 for assert

//#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScanStep(
            int n,
            int d_phase,
            const int* dev_buf_0, 
            int* dev_buf_1) {

            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            if (k >= d_phase) {
                dev_buf_1[k] = dev_buf_0[k - d_phase] + dev_buf_0[k];
            }
            else {
                dev_buf_1[k] = dev_buf_0[k];
            }
            return;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //
            if (n == 0) {
                return;
            }
            assert(odata != nullptr);
            assert(odata != nullptr);
                
            int* device_buf_0;
            int* device_buf_1;
            cudaMalloc((void**)&device_buf_0, n * sizeof(int));
            cudaMalloc((void**)&device_buf_1, n * sizeof(int));
            
            //int* device_idata;
            cudaMemcpy(device_buf_0, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_buf_1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            
            // TODO
            // to device
            int it_ceil = ilog2ceil(n);
            dim3 blocksPerGrid = (n + blocksize - 1) / blocksize;

            int offset;
            for (int d = 1; d <= it_ceil; d++) {
                offset = (int)std::pow(2, d - 1);
                StreamCompaction::Naive::kernScanStep<<<blocksPerGrid, blocksize>>>(n, offset, device_buf_0, device_buf_1);
                //cudaMemcpy(device_buf_0, device_buf_1, n* sizeof(int), cudaMemcpyDeviceToDevice);
                std::swap(device_buf_0, device_buf_1);
            }
            

            timer().endGpuTimer();
            cudaThreadSynchronize();
            // to exclusive
            cudaMemcpy(odata + 1, device_buf_0, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(odata, device_buf_0, n * sizeof(int), cudaMemcpyDeviceToHost);
            /*std::memmove(odata + 1, odata, n-1);
            odata[0] = 0;*/
            cudaFree(device_buf_0);
            cudaFree(device_buf_1);
        }

    }
}

__global__ void scan(float* g_odata, float* g_idata, int n) {
    extern __shared__ float temp[]; // allocated on invocation    
    int thid = threadIdx.x;
    int pout = 0, pin = 1;   // Load input into shared memory.    
                             // This is exclusive scan, so shift right by one    
                             // and set first element to 0   
    temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        // swap double buffer indices     
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * n + thid] += temp[pin * n + thid - offset];
        else
            temp[pout * n + thid] = temp[pin * n + thid];
        __syncthreads();
    }
    g_odata[thid] = temp[pout * n + thid]; // write output 
} 
