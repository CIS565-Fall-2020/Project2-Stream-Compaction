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
        void scan(int N, int *odata, const int *idata) {
            if (N == 0) {
                return;
            }
            assert(odata != nullptr);
            assert(idata != nullptr);
            int* dev_idata; int* dev_odata;

            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            /*thrust::host_vector<int> thrust_hst_odata_vec = thrust*/

            cudaMemcpy(dev_idata, idata, N * sizeof(int), cudaMemcpyHostToDevice);

            thrust::device_ptr<int> dev_thrust_idata = thrust::device_pointer_cast(dev_idata);
            //thrust::device_vector< int > dev_thrust_idata_vec(idata, idata + N);
            thrust::device_vector<int> dev_thrust_idata_vec(dev_thrust_idata, dev_thrust_idata + N);

            thrust::device_ptr<int> dev_thrust_odata = thrust::device_pointer_cast(dev_odata);
            thrust::device_vector<int> dev_thrust_odata_vec(dev_thrust_odata, dev_thrust_odata + N);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_thrust_idata_vec.begin(), dev_thrust_idata_vec.end(), dev_thrust_odata_vec.begin());
            timer().endGpuTimer();

            thrust::copy(dev_thrust_odata_vec.begin(), dev_thrust_odata_vec.end(), odata);
            /*cudaFree(dev_idata);
            cudaFree(dev_odata);*/
        }
    }
}
