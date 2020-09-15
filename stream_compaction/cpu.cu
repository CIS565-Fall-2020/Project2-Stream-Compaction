#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <cassert> // Jack12 for assert
#include <vector> // Jack12 uses vector 

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
        inline void cpu_scan(const int& n, int* odata, const int* idata) {
            // scan without cputimer inline
            if (n == 0) {
                return;
            }

            assert(odata != nullptr);
            assert(idata != nullptr);

            int prefix_sum = 0;

            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                prefix_sum += idata[i - 1];
                odata[i] = prefix_sum;
            }
        }

        void scan(int n, int *odata, const int *idata) {
            
            timer().startCpuTimer();
            // TODO
            // should be exclusive
            cpu_scan(n, odata, idata);
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
            if (n != 0) {
                assert(odata != nullptr);
                assert(idata != nullptr);
            }
            int p = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[p] = idata[i];
                    p++;
                }
            }
            timer().endCpuTimer();
            return p;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int out = 0;
            if (n == 0) {
                return out;
            }
           
            assert(odata != nullptr);
            assert(idata != nullptr);
            
            // map to 0, 1
            int* bin_arr = new int[n];
            for (int i = 0; i < n; i++) {
                bin_arr[i] = idata[i] == 0 ? 0 : 1;
            }
            // scan
            int* scan_arr = new int[n];
            cpu_scan(n, scan_arr, bin_arr);
            // odata contains the scan result
            
            for (int i = 0; i < n; i++) {
                if (bin_arr[i]) {
                    out++;
                    odata[scan_arr[i]] = idata[i];
                }
            }

            delete [] bin_arr;
            delete [] scan_arr;
            timer().endCpuTimer();
            
            return out;
        }
    }
}
