#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"
#include "common.h"

int* dev_radix_data_array, * dev_b_array, * dev_e_array, * dev_f_array, * dev_t_array, * dev_d_array, * dev_output_array;


namespace StreamCompaction {
	namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void compute_b_e(const int n, int* data_array, int* b_array, int* e_array, const int pass) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            // int curr_bit = pow(2.0, pass);
            int curr_bit = 1 << pass;
            int bit_res = curr_bit & data_array[index];
            if (curr_bit == bit_res) {
                // Current bit is 1:
                b_array[index] = 1;
                e_array[index] = 0;
            }
            else {
                // Current bit is 0:
                b_array[index] = 0;

                e_array[index] = 1;
            }
        }

        __global__ void compute_t(const int n, const int total_false, int* f_array, int* t_array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            t_array[index] = index - f_array[index] + total_false;
        }

        __global__ void compute_d(const int n, int* f_array, int* t_array, int* b_array, int* d_array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (b_array[index] > 0) {
                d_array[index] = t_array[index];
            }
            else {
                d_array[index] = f_array[index];
            }
        }

        __global__ void scatter(const int n, int* data_array, int* d_array, int* output_array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int rearranged_index = d_array[index];
            output_array[rearranged_index] = data_array[index];
        }

        void radix_sort(int n, int* odata, const int* idata) {
            int blockSize = 32;
            // dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGrid((n / blockSize) + 1);
            int sizeInBytes = n * sizeof(int);

            // Scan Init:
            int n_ilog2 = ilog2ceil(n);
            // int fit_size = pow(2, n_ilog2);
            int fit_size = 1 << n_ilog2;
            int fitSizeInBytes = fit_size * sizeof(int);
            // dim3 scanFullBlocksPerGrid((fit_size + blockSize - 1) / blockSize);
            dim3 scanFullBlocksPerGrid((n / blockSize) + 1);

            cudaMalloc((void**)&dev_radix_data_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_data_array failed!");
            cudaMalloc((void**)&dev_b_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_b_array failed!");
            cudaMalloc((void**)&dev_e_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_e_array failed!");

            cudaMalloc((void**)&dev_f_array, fitSizeInBytes);
            checkCUDAError("cudaMalloc dev_f_array failed!");

            cudaMalloc((void**)&dev_t_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_t_array failed!");
            cudaMalloc((void**)&dev_d_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_d_array failed!");
            cudaMalloc((void**)&dev_output_array, sizeInBytes);
            checkCUDAError("cudaMalloc dev_output_array failed!");

            cudaMemcpy(dev_radix_data_array, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data_array failed!");

            timer().startGpuTimer();
            int bit_num = sizeof(int) * 8;
            // bit_num = 1;
            for (int pass = 0; pass < bit_num; ++pass) {
                // Compute b:
                // Compute e:
                compute_b_e <<<fullBlocksPerGrid, blockSize>>> (n, dev_radix_data_array, dev_b_array, dev_e_array, pass);
                // Compute f:
                // Scan:
                // Add zeros at the end of array.
                StreamCompaction::Efficient::init_array <<<scanFullBlocksPerGrid, blockSize>>> (dev_f_array, dev_e_array, n, fit_size);
                int d_max = ilog2ceil(n) - 1;
                // Up sweep:
                for (int d = 0; d <= d_max; ++d) {
                    int deno = 1 << (d + 1);
                    // int threads_num_needed = fit_size * pow(0.5, d + 1);
                    int threads_num_needed = fit_size / deno;
                    dim3 up_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                    StreamCompaction::Efficient::up_sweep <<<up_sweep_blocks_per_grid, blockSize>>> (dev_f_array, fit_size, d);
                }
                cudaMemset(dev_f_array + fit_size - 1, 0, sizeof(int));
                // Down sweep:
                for (int d = d_max; d >= 0; --d) {
                    int deno = 1 << (d + 1);
                    // int threads_num_needed = fit_size * pow(0.5, d + 1);
                    int threads_num_needed = fit_size / deno;
                    dim3 down_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                    StreamCompaction::Efficient::down_sweep <<<down_sweep_blocks_per_grid, blockSize>>> (dev_f_array, fit_size, d);
                }
                int e_end, f_end;
                cudaMemcpy(&e_end, dev_e_array + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_e_array failed!");
                cudaMemcpy(&f_end, dev_f_array + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_f_array failed!");
                int total_falses = e_end + f_end;
                // Compute t:
                compute_t << <fullBlocksPerGrid, blockSize >> > (n, total_falses, dev_f_array, dev_t_array);
                checkCUDAError("compute_t failed!");
                // Compute d:
                compute_d << <fullBlocksPerGrid, blockSize >> > (n, dev_f_array, dev_t_array, dev_b_array, dev_d_array);
                checkCUDAError("compute_d failed!");
                // Scatter data for this pass:
                scatter << <fullBlocksPerGrid, blockSize >> > (n, dev_radix_data_array, dev_d_array, dev_output_array);
                checkCUDAError("scatter failed!");
                cudaMemcpy(dev_radix_data_array, dev_output_array, sizeInBytes, cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy output to radix failed!");
            }
            timer().endGpuTimer();

            // Copy to output data:
            cudaMemcpy(odata, dev_output_array, sizeInBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");            

            cudaFree(dev_radix_data_array);
            checkCUDAError("cudaFree(dev_data_array) failed!");
            cudaFree(dev_output_array);
            checkCUDAError("cudaFree(dev_output_array) failed!");
            cudaFree(dev_b_array);
            checkCUDAError("cudaFree(dev_b_array) failed!");
            cudaFree(dev_e_array);
            checkCUDAError("cudaFree(dev_e_array) failed!");
            cudaFree(dev_f_array);
            checkCUDAError("cudaFree(dev_f_array) failed!");
            cudaFree(dev_d_array);
            checkCUDAError("cudaFree(dev_d_array) failed!");
        }
	}
}