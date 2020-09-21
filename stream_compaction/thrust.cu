#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

int* dev_inData;
int* dev_outData;

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            cudaMalloc((void**)&dev_inData, sizeof(int) * n);
            cudaMalloc((void**)&dev_outData, sizeof(int) * n);
            cudaMemcpy(dev_inData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            thrust::device_ptr<int> dev_thrust_idata(dev_inData);
            thrust::device_ptr<int> dev_thrust_odata(dev_outData);
            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_outData, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_inData);
            cudaFree(dev_outData);
        }
    }
}
