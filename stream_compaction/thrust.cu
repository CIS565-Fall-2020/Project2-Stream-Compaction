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
			//thrust::host_vector<int> host_data(n);
			//thrust::copy(idata, idata + n, host_data.begin());
			//thrust::device_vector<int> dev_idata = host_data;
			//thrust::device_vector<int> dev_odata(n);
            timer().startGpuTimer();
            // DONE use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            //thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());
			thrust::exclusive_scan(idata, idata + n, odata);
            timer().endGpuTimer();
			//host_data = dev_odata;
			//thrust::copy(host_data.begin(), host_data.begin() + n, odata);
        }
    }
}
