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

		// GPU Gems 3 example
		__global__ void prescan(float *g_odata, float *g_idata, int n) {
			extern __shared__ float temp[];  // allocated on invocation 
			int thid = threadIdx.x;
			int offset = 1;
			temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory 
			temp[2 * thid + 1] = g_idata[2 * thid + 1];
			for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree 
			{
				__syncthreads();
				if (thid < d) {
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
				if (thid < d) {
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					float t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();
			g_odata[2 * thid] = temp[2 * thid]; // write results to device memory      
			g_odata[2 * thid + 1] = temp[2 * thid + 1];
		}

		__global__ void kernelEfficientScan(int *g_odata, int *g_idata, int n, int N) {
			int index = threadIdx.x;
			int offset = 2;
			g_odata[index] = g_idata[index];
			// up-sweep
			for (int d = N / 2; d >= 1; d >>= 1) {
				__syncthreads();
				if (index < d) {
					int a = n - 1 - (index * offset);
					int b = a - offset / 2;
					if (a >= 0 && b >= 0) {
						g_odata[a] += g_odata[b];
					}
				}
				offset *= 2;
			}
			// down-sweep
			if (index == 0 && n > 0) {
				g_odata[n - 1] = 0;
			}
			offset /= 2;
			for (int d = 1; d <= N / 2; d *= 2) {
				__syncthreads();
				if (index < d) {
					int a = n - 1 - (index * offset);
					int b = a - offset / 2;
					if (a >= 0 && b >= 0) {
						int tmp = g_odata[b];
						g_odata[b] = g_odata[a];
						g_odata[a] += tmp;
					}
				}
				offset /= 2;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int k = ilog2ceil(n);
			int N = (int) pow(2, k);
			
			int *g_odata;
			int *g_idata;
			cudaMalloc((void**)&g_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc g_idata failed!");
			cudaMalloc((void**)&g_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc g_odata failed!");
			cudaMemcpy(g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
			kernelEfficientScan<<<1, n >>>(g_odata, g_idata, n, N);

            timer().endGpuTimer();

			// copy back ouput
			cudaMemcpy(odata, g_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy odata failed!");

			cudaFree(g_odata);
			cudaFree(g_idata);
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
