#pragma once

#include "common.h"

constexpr int efficient_blocksize = 128;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool ifTimer, bool ifIdxScale, bool ifSharedMemory);

        int compact(int n, int *odata, const int *idata);
    }
}
