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
            int j = 0;
            for (int i = 0; i < n; i++) {
              if (idata[i] != 0) {
                odata[j++] = idata[i];
              }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *temp = new int[n],
                *tempSum = new int[n];
            timer().startCpuTimer();
            for (int i = 0; i < n; i++) {
              temp[i] = idata[i] == 0 ? 0 : 1;
            }
            tempSum[0] = 0;
            for (int i = 1; i < n; i++) {
                tempSum[i] = tempSum[i - 1] + temp[i - 1];
            }
            int cnt = 0;
            for (int i = 0; i < n; i++) {
              if (temp[i] == 1) {
                odata[tempSum[i]] = idata[i];
                cnt++;
              }
            }
            timer().endCpuTimer();
            delete[] temp;
            delete[] tempSum;
            return cnt;
        }
    }
}
