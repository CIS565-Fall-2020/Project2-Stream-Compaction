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
            // TODO -> DONE
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
            // TODO -> DONE

            int num = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[num] = idata[i];
                    num++;
                }
            }
            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO -> DONE
            const int size = n;
            int* temp = new int[size];

            //mapping 
            for (int i = 0; i < n; i++) {
                temp[i] = (idata[i] != 0) ? 1 : 0;
            }

            // scanning
            int* scannedArray = new int[size];
            scannedArray[0] = 0;
            for (int i = 1; i < n; i++) {
                scannedArray[i] = scannedArray[i - 1] + temp[i - 1];
            }

            // Scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (temp[i] == 1) {
                    odata[scannedArray[i]] = idata[i];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }
    }
}
