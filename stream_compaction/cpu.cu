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
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
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
            int ctr = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[ctr] = idata[i];
                    ctr++;
                }
            }
            timer().endCpuTimer();
            return ctr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            
            int ctr = 0;
            int* marker = new int[n];
            int* scan_res = new int[n];

            for (int i = 0; i < n; i++) {
                scan_res[i] = 0;
                marker[i] = 0;
            }

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    marker[i] = 1;
                }
            }
            
            for (int i = 1; i < n; i++) {
                scan_res[i] = marker[i-1] + scan_res[i-1];
            }

            for (int i = 0; i < n; i++) {
                if (marker[i] == 1) {
                    odata[scan_res[i]] = idata[i];
                    ctr++;
                 }
            }
            
            delete[] scan_res;
            delete[] marker;
            
            timer().endCpuTimer();
            
            return ctr;
        }
    }
}
