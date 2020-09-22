#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_idata;
        int* dev_odata;

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            thrust::device_ptr<int> dv_idata_ptr(dev_idata);
            thrust::device_ptr<int> dv_odata_ptr(dev_odata);

            const int blockSize = 64;
            dim3 numBoidBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            thrust::exclusive_scan(dv_idata_ptr, dv_idata_ptr + n, dv_odata_ptr);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

        }
    }
}
