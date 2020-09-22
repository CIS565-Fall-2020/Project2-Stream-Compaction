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

            timer().startGpuTimer();
            thrust::host_vector<int> host_v(n);
            thrust::copy(idata, idata + n, host_v.begin());
            thrust::device_vector<int> dev_v = host_v;
            thrust::device_vector<int> out_v(n);
            thrust::exclusive_scan(dev_v.begin(), dev_v.end(), out_v.begin());
            thrust::copy(out_v.begin(), out_v.end(), odata);
            timer().endGpuTimer();
        }
    }
}
