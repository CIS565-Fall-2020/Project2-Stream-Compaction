#pragma once

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radixSort(int n, int bits_num, int* odata, const int* idata);
    }
}