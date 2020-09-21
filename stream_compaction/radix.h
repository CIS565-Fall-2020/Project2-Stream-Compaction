#pragma once

#include "common.h"

namespace StreamCompaction {
	namespace RadixSort {
		StreamCompaction::Common::PerformanceTimer& timer();

		void radix_sort(int n, int* odata, const int* idata);
	}
}
