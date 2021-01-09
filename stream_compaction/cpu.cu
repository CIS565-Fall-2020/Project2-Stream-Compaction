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
            int sum = 0; // identity
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
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
            int p = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) { // remove 0
                    odata[p++] = idata[i];
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
            if (n < 1) {
                return 0;
            }
            int* e = new int[n];
            for (int i = 0; i < n; i++) {
                e[i] = idata[i] != 0;
            }

            // scan
            int* prefix = new int[n];
            int sum = 0;
            for (int i = 0; i < n; i++) {
                prefix[i] = sum;
                sum += e[i];
            }

            // scatter
            for (int i = 0; i < n - 1; i++) {
                if (prefix[i] != prefix[i + 1]) {
                    odata[prefix[i]] = idata[i];
                }
            }
            int len = prefix[n - 1];
            if (e[n - 1] == 1) {
                len++;
                odata[prefix[n - 1]] = idata[n - 1];
            }
            delete[] e;
            delete[] prefix;
            timer().endCpuTimer();
            return len;
        }
    }
}
