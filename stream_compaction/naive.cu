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

        int* dev_bufferA;
        int* dev_bufferB;
        int numObjects;
        
        __global__ void kernNaiveScan(int N, int* A, int* B, int temp) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) {
                return;
            }
            if (index < temp) {
                A[index] = B[index];
                return;
            }
            A[index] = B[index - temp] + B[index];
        }

        void initSimulation(int N, const int* B) {
            numObjects = N;
            cudaMalloc((void**)&dev_bufferA, N * sizeof(int));
            cudaMalloc((void**)&dev_bufferB, N * sizeof(int));
            cudaMemcpy(dev_bufferB, B, N * sizeof(int), cudaMemcpyHostToDevice);
        }

        void endSimulation() {
            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            initSimulation(n, idata);
            const int blockSize = 256;
            dim3 numBoidBlocks((n + blockSize - 1) / blockSize);
            int dmax = ilog2ceil(n);
            
            timer().startGpuTimer();

            for (int i = 1; i <= dmax; i++) {
                kernNaiveScan << <numBoidBlocks, blockSize >> > (n, dev_bufferA, dev_bufferB, int(powf(2, i - 1)));
                std::swap(dev_bufferA, dev_bufferB);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_bufferB, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            odata[0] = 0;
            endSimulation();
        }
    }
}
