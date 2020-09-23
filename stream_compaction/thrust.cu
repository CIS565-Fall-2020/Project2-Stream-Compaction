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
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            //DONE
            thrust::host_vector<int> hostVec(n);
            thrust::copy(idata, idata + n, hostVec.begin());
            thrust::device_vector<int> devVec = hostVec;
            thrust::device_vector<int> outVec(n);
            thrust::exclusive_scan(devVec.begin(), devVec.end(), outVec.begin());
            thrust::copy(outVec.begin(), outVec.end(), odata);
            timer().endGpuTimer();
        }
    }
}
