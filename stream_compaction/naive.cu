#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(const int n, const int minIdx, int *odata, const int *idata) {
            int idx = threadIdx.x +blockDim.x * blockIdx.x;
            if (idx >= n) { return; }

            int right_parent = idata[idx];
            int left_parent = (idx < minIdx) ? 0 : idata[idx - minIdx];
            odata[idx] = right_parent + left_parent;
        }

        __global__ void kernShift(const int n, int* odata, const int* idata) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= n) { return; }
            odata[idx] = (idx == 0) ? 0 : idata[idx - 1];
        }
        
        void cudaScanNaive(int n, int* &GPUout, int* &GPUin) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            for (int i = 1; i < n; i = i << 1) {
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, i, GPUout, GPUin);
                int* tmp = GPUin;
                GPUin = GPUout;
                GPUout = tmp;
            }

            kernShift << <fullBlocksPerGrid, blockSize >> > (n, GPUout, GPUin);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            
            int* GPUin;
            int* GPUout;
            cudaMalloc((void**)&GPUin, n*sizeof(int));
            cudaMalloc((void**)&GPUout, n*sizeof(int));
            cudaMemcpy(GPUin, idata, n*sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            cudaScanNaive(n, GPUout, GPUin);
            timer().endGpuTimer();

            cudaMemcpy(odata, GPUout, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(GPUin);
            cudaFree(GPUout);
            
        }
    }
}
