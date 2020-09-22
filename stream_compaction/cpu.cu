#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <vector>

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
            // TODO (exclusive)
			if (n > 0) {
				odata[0] = 0;
				for (int i = 1; i < n; i++) {
					odata[i] = odata[i - 1] + idata[i - 1];
				}
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
			int idx = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[idx] = idata[i];
					idx++;
				}
			}
            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
			std::vector<int> tmp(n, 0);
			std::vector<int> scan_result(n);
			int count = 0;
			// build tmp binary array
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					tmp[i] = 1;
					count++;
				}
			}
			// scan
			if (n > 0) {
				scan_result[0] = 0;
				for (int k = 1; k < n; k++) {
					scan_result[k] = scan_result[k - 1] + tmp[k - 1];
				}
			}
			// scatter
			for (int i = 0; i < n; i++) {
				if (tmp[i] == 1) {
					int idx = scan_result[i];
					odata[idx] = idata[i];
				}
			}
            timer().endCpuTimer();
            return count;
        }
    }
}
