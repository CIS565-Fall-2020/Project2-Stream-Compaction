#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void init_array(int* dev_array, const int* dev_temp_array, const int n, const int fit_size);

        __global__ void up_sweep(int* dev_array, const int fit_size);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
