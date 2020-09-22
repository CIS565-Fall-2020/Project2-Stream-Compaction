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

        void CPUScan(const int* idata, int* odata, int n)
        {
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        int CPUCompactWithoutScan(int n, int* odata, const int* idata)
        {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            return count;
        }
        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            CPUScan(idata, odata, n);
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
            int count = CPUCompactWithoutScan(n, odata, idata);
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {

            int* boolArray = (int*)malloc(n * sizeof(int));
            int* indexArray = (int*)malloc(n * sizeof(int));

            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; ++i) {
                boolArray[i] = (idata[i]) ? 1 : 0;
            }

            // scan
            CPUScan(boolArray, indexArray, n);

            // scatter

            for (int i = 0; i < n; ++i) {
                if (boolArray[i]) {
                    odata[indexArray[i]] = idata[i];
                }
            }
            int count = boolArray[n - 1] ? indexArray[n - 1] + 1 : indexArray[n - 1];
            timer().endCpuTimer();
            free(boolArray);
            free(indexArray);
            return count;
        }
    }
}
