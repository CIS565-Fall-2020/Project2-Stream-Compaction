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

            int prefixSum = 0;
            for (int i = 0; i < n; i++)
            {
                odata[i] = prefixSum;
                prefixSum += idata[i];
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
            
            int cnt = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                    odata[cnt++] = idata[i];
            }

            timer().endCpuTimer();
            return cnt;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Compute temporary array
            int* tdata = new int[n];
            for (int i = 0; i < n; i++)
            {
                tdata[i] = idata[i] != 0 ? 1 : 0;
            }

            // Run exclusive scan on temporary array
            int* sdata = new int[n];
            scan(n, sdata, tdata);

            // Scatter
            int idx = 0;
            for (int i = 0; i < n; i++)
            {
                if (tdata[i])
                {
                    idx = sdata[i];
                    odata[idx] = idata[i];
                }
            }
            timer().endCpuTimer();
            return idx + 1;
        }
    }
}
