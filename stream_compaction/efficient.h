#pragma once

#include "common.h"

constexpr int efficient_blocksize = 256;

enum class EFF_method {
    nonOptimization,
    idxMapping,
    sharedMemory
};

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, EFF_method cur_method, bool ifTimer);

        int compact(int n, int *odata, const int *idata, EFF_method cur_method);
    }
}
