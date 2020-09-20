#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernEfficientUpSweep(int n, int offset,
            int numNode, int* data);

        __global__ void kernEfficientDownSweep(int n, int offset,
            int numNode, int* data);

        void scanHelper(int full, int d, int blockSize, int* dev_data);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
