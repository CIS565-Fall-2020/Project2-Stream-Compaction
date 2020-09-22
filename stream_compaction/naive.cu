#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCKSIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScanNaive(int n, int* odata, int* idata, int lowbound)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= lowbound){
                odata[index] = idata[index - lowbound] + idata[index];
            }
            else{
                odata[index] = idata[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy (host to device) failed!");

            int d = ilog2ceil(n);//log2 n
            dim3 fullBlocksPerGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);
            timer().startGpuTimer();
            // TODO
            for (int i = 1; i <= d; i++) {
                if((i % 2))
                {
                    kernScanNaive << <fullBlocksPerGrid, dim3(BLOCKSIZE) >> > (n, dev_odata, dev_idata, 1 << (i - 1));
                }
                else
                {
                    kernScanNaive << <fullBlocksPerGrid, dim3(BLOCKSIZE) >> > (n, dev_idata, dev_odata, 1 << (i - 1));
                }
            }
            timer().endGpuTimer();

            //Decide which is the source of the final scan
            int* source;
            if (d % 2) {
                source = dev_odata;
            }
            else {
                source = dev_idata;
            }
            //To transfer inclusive scan to exclusive scan, do a right shift.
            cudaMemcpy(odata + 1, source, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("get odata failed!");
            odata[0] = 0;
            //free
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
