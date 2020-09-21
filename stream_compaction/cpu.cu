#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int k = 1; k < n; ++k)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int o_idx = 0;
            for (int i_idx = 0; i_idx < n; ++i_idx) {
                if (idata[i_idx] != 0) {
                    odata[o_idx] = idata[i_idx];
                    ++o_idx;
                }
            }
            timer().endCpuTimer();
            return o_idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* temp_array = new int[n];
            int* scan_array = new int[n];
            // Compute temporary array:
            for (int i_idx = 0; i_idx < n; ++i_idx) {
                if (idata[i_idx] != 0) {
                    temp_array[i_idx] = 1;
                }
                else {
                    temp_array[i_idx] = 0;
                }
            }
            // Exclusive scan:
            scan_array[0] = 0;
            for (int k = 1; k < n; ++k)
            {
                scan_array[k] = scan_array[k - 1] + temp_array[k - 1];
            }
            // Scatter:
            int o_counter = 0;
            for (int i_idx = 0; i_idx < n; ++i_idx) 
            {
                if (temp_array[i_idx] == 1) {
                    int o_idx = scan_array[i_idx];
                    odata[o_idx] = idata[i_idx];
                    ++o_counter;
                }
            }
            timer().endCpuTimer();
            return o_counter;
        }
    }
}
