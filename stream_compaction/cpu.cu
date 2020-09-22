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
			for (int i = 1; i < n; ++i)
			{
				odata[i] = idata[i - 1] + odata[i - 1];
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
            int oIndex = 0;
            for(int i = 0; i < n; ++i)
            {
                if(idata[i] != 0)
                {
                    odata[oIndex] = idata[i];
                    oIndex ++;
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
            // TODO
            int *mappedArr = new int[n];
            int *scannedArr = new int[n];
            

            // Compute temporary array containing 1 and 0
            for(int i = 0; i < n; ++i)
            {
                if(idata[i] != 0)
                {   
                    mappedArr[i] = 1;
                }
                else 
                {
                    mappedArr[i] = 0;
                }
            }

            // Run exclusive scan on mapped array
            scan(n, scannedArr, mappedArr);

            // Scatter
            int oCount = 0;
            for(int i = 0; i < n; ++i)
            {
                if(mappedArr[i] != 0)
                {  
                    int index = scannedArr[i];
                    odata[index] = idata[i];
                    oCount ++;
                }
            }
            
            timer().endCpuTimer();
            return oCount;
        }
    }
}
