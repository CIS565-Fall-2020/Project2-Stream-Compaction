#include "efficient.h"
#include <device_launch_parameters.h>
#include <cassert> 
#include "radixSort.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace StreamCompaction{
	namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void CpuStandardSort(const int& N, int* out, const int* in) {
            if (N == 0) {
                return;
            }
            assert(in != nullptr);
            assert(out != nullptr);

            std::vector<int> a_vec(in, in + N);

            timer().startCpuTimer();
            std::sort(a_vec.begin(), a_vec.end());
            timer().endCpuTimer();

            std::copy(a_vec.begin(), a_vec.end(), out);
        }

        /*inline void GpuradixSort(const int& N, )*/
	}
}
