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
            thrust::device_vector<int> dev_vec(idata, idata + n), dev_out(n);

            timer().startGpuTimer();
            // use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_vec.begin(), dev_vec.end(), dev_out.begin());
            timer().endGpuTimer();

            thrust::copy(dev_out.begin(), dev_out.end(), odata);
        }

        struct isNonZero {
            __host__ __device__ bool operator()(int x) {
                return x != 0;
            }
        };

        int compact(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dev_vec(idata, idata + n), dev_out(n);

            timer().startGpuTimer();
            int count = thrust::copy_if(
                dev_vec.begin(), dev_vec.end(), dev_out.begin(), isNonZero()
            ) - dev_out.begin();
            timer().endGpuTimer();

            thrust::copy(dev_out.begin(), dev_out.end(), odata);
            return count;
        }
    }
}
