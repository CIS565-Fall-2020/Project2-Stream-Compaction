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
		void scan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			//Exclusive Scan 
			if (odata && idata && n > 0)
			{
				odata[0] = 0;
				for (int i = 1; i < n; ++i)
				{
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
		int compactWithoutScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			int count = 0;
			if (odata && idata && n > 0)
			{
				for (int i = 0; i < n; ++i)
				{
					if (idata[i] != 0)
					{
						odata[count] = idata[i];
						count++;
					}
				}
			}
			timer().endCpuTimer();
			return count;
		}

		//Scan Helper function for compactWithScan
		void scanHelper(int n, int* odata, const int* idata) {
			// TODO
			//Exclusive Scan 
			if (odata && idata && n > 0)
			{
				odata[0] = 0;
				for (int i = 1; i < n; ++i)
				{
					odata[i] = odata[i - 1] + idata[i - 1];
				}
			}
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();

			// TODO        
			int count = 0;
			int* tempArray = new int[n];
			int* scanResult = new int[n];

			if (odata && idata && n > 0)
			{
				//Compute temp array with 1s and 0s 
				for (int i = 0; i < n; ++i)
				{
					if (idata[i])
					{
						tempArray[i] = 1;
					}
					else
					{
						tempArray[i] = 0;
					}
				}

				//Scan           
				scanHelper(n, scanResult, tempArray);


				//printf("CPU Steam Compaction scanResult Output is [");
				//for (int i = 0; i < count; i++)
				//{
				//	printf(" %d  ", scanResult[i]);
				//}
				//printf("] \n");

				//Scatter              
				for (int i = 0; i < n; ++i)
				{
					if (tempArray[i] == 1)
					{
						odata[scanResult[i]] = idata[i];
						count++;
					}
				}

				//printf("CPU SC Count is %d \n", count);
				//printf("CPU Steam Compaction Output is [");
				//for (int i = 0; i < count; i++)
				//{
				//        printf(" %d  ", odata[i]);                   
				//}
				//printf("] \n");
			}
			timer().endCpuTimer();

			delete[] tempArray;
			delete[] scanResult;

			return count++;
		}
	}
}
