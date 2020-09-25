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

        // used help from: https://docs.nvidia.com/cuda/thrust/index.html#:~:text=As%20the%20names%20suggest%2C%20host_vector%20is%20stored%20in,any%20data%20type%29%20that%20can%20be%20resized%20dynamically.
        void scan(int n, int *odata, const int *idata) {
            
            // create a host vector
            thrust::host_vector<int> host_vector(n);
            // initialize individual elements
            thrust::copy(idata, idata + n, host_vector.begin());

            // create a device vector from host vector
            thrust::device_vector<int> dv_in = host_vector;
            thrust::device_vector<int> dv_out(n);

            timer().startGpuTimer();

            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            timer().endGpuTimer();

            thrust::copy(dv_out.begin(), dv_out.end(), odata);
            
        }
    }
}
