#pragma once
#include "common.h"
constexpr int radix_blocksize = 256;

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();
        
        void CpuStandardSort(const int& N, int* out, const int* in);
    }
}