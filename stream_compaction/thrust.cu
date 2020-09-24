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
            thrust::host_vector<int> h_out(n);
            thrust::host_vector<int> h_in(n);
            for (int i = 0; i < n; i++) {
                h_in[i] = idata[i];
                h_out[i] = idata[i];
            }


            thrust::device_vector<int> dev_out = h_out;
            thrust::device_vector<int> dev_in = h_in;

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();

            thrust::copy(dev_out.begin(), dev_out.end(), h_out.begin());
            for (int i = 0; i < n; i++) {
                odata[i] = h_out[i];
            }

        }
    }
}
