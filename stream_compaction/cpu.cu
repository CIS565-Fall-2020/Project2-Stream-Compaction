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

        void cpu_scan(int n, int* odata, const int* idata) {
            for (int i = 0; i < n; i++) {
                if (i == 0) { odata[i] = 0; continue; }
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
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
            int pointer = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[pointer] = idata[i];
                    pointer++;
                }
            }
            timer().endCpuTimer();
            return pointer;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int* binary = new int[n];
            int* scanned = new int[n];

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    binary[i] = 1;
                }
                else {
                    binary[i] = 0;
                }
            }

            cpu_scan(n, scanned, binary);
            int lastPointer = 0;
            for (int i = 0; i < n; i++) {
                if (scanned[i] > lastPointer) {
                    odata[lastPointer] = idata[i-1];
                    lastPointer++;
                }
            }

            if (idata[n - 1] != 0) {
                odata[lastPointer] = idata[n - 1];
                lastPointer++;
            }

            delete[n] binary;
            delete[n] scanned;

            timer().endCpuTimer();
            return lastPointer;
        }
    }


}
