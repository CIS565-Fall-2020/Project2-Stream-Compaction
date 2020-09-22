#include "efficient.h"
#include <device_launch_parameters.h>
#include <cassert> 
#include "radixSort.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

int k_th_bit(int k, int n) {
    return (n >> k) & 1;
}

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void CpuStandardSort(const int& N, int* out, const int* in) {
            if (N == 0) {
                return;
            }
            assert(in != nullptr);
            assert(out != nullptr);

            std::vector<int> a_vec(in, in + N);

            timer().startCpuTimer();
            std::sort(a_vec.begin(), a_vec.end());
            timer().endCpuTimer();

            std::copy(a_vec.begin(), a_vec.end(), out);
        }

        void GpuRadixSort(const int& N, int* hst_out, const int* hst_in, const int max_bit ){
            //
            if (N == 0) {
                return;
            }
            assert(hst_in != nullptr);
            assert(hst_out != nullptr);

            /*int* dev_in, dev_out, dev_out_buf;
            cudaMalloc((void**)&dev_in, N * sizeof(int));
            cudaMalloc((void**)&dev_out, N * sizeof(int));
            cudaMalloc((void**)&dev_out_buf, N * sizeof(int));
            cudaMemcpy(dev_in, hst_in, N * sizeof(int), cudaMemcpyHostToDevice);*/

            int* hst_e,* hst_f,* hst_d;
            int* hst_out_buf;
            hst_e = new int[N];
            hst_f = new int[N];
            hst_d = new int[N];

            hst_out_buf = new int[N];
            std::copy(hst_in, hst_in + N, hst_out_buf);

            timer().startGpuTimer();
            for (int k = 0; k < max_bit; k ++) {
                for (int i = 0; i < N; i++) {
                    hst_e[i] = 1 - k_th_bit(k, hst_out_buf[i]);
                }

                Efficient::scan(N, hst_f, hst_e, false, false, true);

                int total_falses = hst_e[N - 1] + hst_f[N - 1];
                
                for (int i = 0; i < N; i++) {
                    hst_d[i] = hst_e[i] == 0 ? (i - hst_f[i] + total_falses) : hst_f[i];
                }

                for (int i = 0; i < N; i++) {
                    hst_out[hst_d[i]] = hst_out_buf[i];
                }
                std::copy(hst_out, hst_out + N, hst_out_buf);
            }

            timer().endGpuTimer();

            delete[] hst_e;
            delete[] hst_f;
            delete[] hst_d;
            delete[] hst_out_buf;
        }
	}
}
