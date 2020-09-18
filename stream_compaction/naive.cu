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
		__global__ void kernelScan(int *g_odata, int *g_idata,  int *A_data, int *B_data, int k) {   
			/*
			extern __shared__ float temp[]; // allocated on invocation    
			int thid = threadIdx.x;   
			int pout = 0, pin = 1;   // Load input into shared memory.    
									 // This is exclusive scan, so shift right by one   
									 // and set first element to 0   
			temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;   
			__syncthreads();   
			for (int offset = 1; offset < n; offset *= 2)   {     
				pout = 1 - pout; // swap double buffer indices     
				pin = 1 - pout;     
				if (thid >= offset)       
					temp[pout*n+thid] += temp[pin*n+thid - offset];     
				else       temp[pout*n+thid] = temp[pin*n+thid];     
				__syncthreads();   
			}   
			g_odata[thid] = temp[pout*n+thid]; // write output 
			*/

			int index = threadIdx.x;
			A_data[index] = (index > 0) ? g_idata[index - 1] : 0;
			__syncthreads();
			int offset = 1;
			for (int d = 1; d <= k; d++) {
				if (index >= offset) {
					B_data[index] = A_data[index] + A_data[index - offset];
				}
				else {
					B_data[index] = A_data[index];
				}
				offset *= 2;
				__syncthreads();
				// swap pointers
				int *tmp = A_data;
				A_data = B_data;
				B_data = tmp;
			}
			// point odata to A_data
			g_odata[index] = A_data[index];
		} 


        /*** 
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// allocate device arrays
			int *A_data;
			int *B_data;
			cudaMalloc((void**)&A_data, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc A_data failed!");
			cudaMalloc((void**)&B_data, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc B_data failed!");

			int *g_odata;
			int *g_idata;
			cudaMalloc((void**)&g_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc g_idata failed!");
			cudaMalloc((void**)&g_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc g_odata failed!");
			// cudaMemcpy(g_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int k = ilog2ceil(n);

            timer().startGpuTimer();
            // TODO
			kernelScan<<<1, n>>>(g_odata, g_idata, A_data, B_data, k);

            timer().endGpuTimer();

			// copy output
			cudaMemcpy(odata, g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy odata failed!");
			// free device arrays
			cudaFree(A_data);
			cudaFree(B_data);
			cudaFree(g_odata);
			cudaFree(g_idata);
        }
    }
}
