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
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_idata, *dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			thrust::device_ptr<int> dev_thrust_odata(dev_odata);
			thrust::device_ptr<int> dev_thrust_idata(dev_idata);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
            timer().endGpuTimer();

			cudaMemcpy(odata, thrust::raw_pointer_cast(dev_thrust_odata), n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
