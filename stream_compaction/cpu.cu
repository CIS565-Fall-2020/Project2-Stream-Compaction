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
			odata[0] = 0;
			for (int k = 1; k < n; ++k) {
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
			// TODO
			int ptr = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[ptr] = idata[i];
					ptr++;
				}
			}
			timer().endCpuTimer();
			return ptr;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();
			// TODO
			int count = 0;
			int *tmp = new int[n];
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					tmp[i] = 0;
				}
				else {
					tmp[i] = 1;
					count++;
				}
			}
			int *tmpScan = new int[n];
			tmpScan[0] = 0;
			for (int k = 1; k < n; ++k) {
				tmpScan[k] = tmpScan[k - 1] + tmp[k - 1];
			}
			for (int i = 0; i < n; i++) {
				odata[tmpScan[i]] = idata[i];
			}
			timer().endCpuTimer();
			delete tmp;
			delete tmpScan;
			return count;
		}
	}
}
