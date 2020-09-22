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
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i];
            }

            // exclusive scan
            odata[0] = 0;

            for (int i = 0; i < n - 1; i++) {
                odata[i + 1] = odata[i] + idata[i];
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
            const int size = n;
            // Map to temp array

            int oIndex = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[oIndex] = idata[i];
                    oIndex++;
                }
            }

            timer().endCpuTimer();
            return oIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            const int size = n;

            // Map to temp array
            int* tempArray = new int[size];
            for (int i = 0; i < n; i++) {
                tempArray[i] = (idata[i] == 0) ? 0 : 1;
            }

            // Exclusive scan
            int* scannedArray = new int[size];
            for (int i = 0; i < n; i++) {
                scannedArray[i] = tempArray[i];
            }
            scannedArray[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                scannedArray[i + 1] = scannedArray[i] + tempArray[i];
            }

            // Scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (tempArray[i] == 1) {
                    odata[scannedArray[i]] = idata[i];
                    count++;
                }
            }
            
            timer().endCpuTimer();
            return count;
        }
    }
}
