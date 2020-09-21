#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        __global__ void addition_process(float* dev_data_array1, float* dev_data_array2, const int d, const int n);
    }
}
