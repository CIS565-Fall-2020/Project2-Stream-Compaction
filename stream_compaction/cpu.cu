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
            // we are assuming that the arrays have at least one element
            timer().startCpuTimer();
            odata[0] = 0;
            for (int k = 1; k < n; k++) {
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
            int optr = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[optr] = idata[i];
                    optr++;
                }
            }
            timer().endCpuTimer();
            return optr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* zeroOnes = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    zeroOnes[i] = 1;
                }
                else {
                    zeroOnes[i] = 0;
                }
            }
            int* scanResult = new int[n];

            // scan
            scanResult[0] = 0;
            for (int k = 1; k < n; k++) {
                scanResult[k] = scanResult[k - 1] + zeroOnes[k - 1];
            }
            // end of scan

            for (int i = 0; i < n; i++) {
                if (zeroOnes[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                }
            }
            int count = scanResult[n - 1];
            delete[] zeroOnes;
            delete[] scanResult;
            timer().endCpuTimer();
            return count;
        }
    }
}
