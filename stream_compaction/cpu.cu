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
            // compute an exclusive prefix sum
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
            int elements_remaining = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[elements_remaining] = idata[i];
                    elements_remaining++;
                }
            }
            timer().endCpuTimer();
            return elements_remaining;
        }

        /*
        * Helper Function because I seem to be having issues when I start the timer again
        * Same as scan function earlier, just without the timer
        * From Piazza @110
        */
        void cpu_scan(int n, int* odata, const int* idata) {
            // compute an exclusive prefix sum
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // fill temp array with 0 if idata is 0 or 1 otherwise
            int* temp_array = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    temp_array[i] = 0;
                }
                else temp_array[i] = 1;
            }
            
            // run exclusive scan on temporary array
            int* scanned_array = new int[n];
            cpu_scan(n, scanned_array, temp_array);

            // scatter
            for (int j = 0; j < n; j++) {
                if (temp_array[j] == 1) {
                    // write element 
                    odata[scanned_array[j]] = idata[j];
                }
            }
            
            // cleanup
            int result = scanned_array[n - 1];
            timer().endCpuTimer();
            delete[] temp_array;
            delete[] scanned_array;
            return result;
        }
    }
}
