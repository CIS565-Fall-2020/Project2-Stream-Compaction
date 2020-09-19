#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

int* dev_data_array;
int* dev_temp_data_array;

int* dev_zero_one_temp_array;
int* dev_idata_array;
int* dev_final_array;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void init_array(int* dev_array, const int* dev_temp_array, const int n, const int fit_size) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= fit_size) {
                return;
            }
            if (index < n) {
                dev_array[index] = dev_temp_array[index];
            }
            else {
                dev_array[index] = 0;
            }
        }

        __global__ void up_sweep(int* dev_array, const int fit_size, const int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int remap_index = index * pow(2.0, d + 1);
            if (remap_index >= fit_size) {
                return;
            }
            int two_pow_d_add_1 = pow(2.0, d + 1);
            int two_pow_d = pow(2.0, d);
            dev_array[remap_index + two_pow_d_add_1 - 1] += dev_array[remap_index + two_pow_d - 1];
        }

        __global__ void down_sweep(int* dev_array, const int fit_size, const int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int remap_index = index * pow(2.0, d + 1);
            if (remap_index >= fit_size) {
                return;
            }
            int two_pow_d_add_1 = pow(2.0, d + 1);
            int two_pow_d = pow(2.0, d);
            int t = dev_array[remap_index + two_pow_d - 1];
            dev_array[remap_index + two_pow_d - 1] = dev_array[remap_index + two_pow_d_add_1 - 1];
            dev_array[remap_index + two_pow_d_add_1 - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Init all the requirement data:
            int n_ilog2 = ilog2ceil(n);
            int fit_size = pow(2, n_ilog2);

            int oriSizeInBytes = n * sizeof(int);
            int fitSizeInBytes = fit_size * sizeof(int);
            int blockSize = 128;
            dim3 fullBlocksPerGrid((fit_size + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_data_array, fitSizeInBytes);
            checkCUDAError("cudaMalloc dev_data_array failed!");
            cudaMalloc((void**)&dev_temp_data_array, oriSizeInBytes);
            checkCUDAError("cudaMalloc dev_temp_data_array failed!");

            cudaMemcpy(dev_temp_data_array, idata, oriSizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data_array failed!");

            init_array <<<fullBlocksPerGrid, blockSize>>> (dev_data_array, dev_temp_data_array, n, fit_size);

            timer().startGpuTimer();
            
            // TODO
            int d_max = ilog2ceil(n) - 1;
            // Up-Sweep:
            for (int d = 0; d <= d_max; ++d) {
                int threads_num_needed = fit_size * pow(0.5, d + 1);
                dim3 up_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                up_sweep <<<up_sweep_blocks_per_grid, blockSize>>> (dev_data_array, fit_size, d);
            }
            
            cudaMemset(dev_data_array + fit_size - 1, 0, sizeof(int));
            // Down-Sweep:
            for (int d = d_max; d >= 0; --d) {
                int threads_num_needed = fit_size * pow(0.5, d + 1);
                dim3 down_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                down_sweep <<<down_sweep_blocks_per_grid, blockSize>>> (dev_data_array, fit_size, d);
            }
            
            // Copy to output data:
            cudaMemcpy(odata, dev_data_array, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            timer().endGpuTimer();
            cudaFree(dev_data_array);
            checkCUDAError("cudaFree(dev_data_array) failed!");
            cudaFree(dev_temp_data_array);
            checkCUDAError("cudaFree(dev_temp_data_array) failed!");
            // free(a);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int sizeInBytes = n * sizeof(int);

            int* zero_one_temp_array = new int[n];
            int* scan_result_array = new int[n];
            cudaMalloc((void**)&dev_zero_one_temp_array, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_zero_one_temp_array failed!");
            cudaMalloc((void**)&dev_idata_array, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata_array failed!");

            cudaMemcpy(dev_zero_one_temp_array, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_zero_one_temp_array failed!");
            cudaMemcpy(dev_idata_array, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata_array failed!");

            // Scan Init:
            int n_ilog2 = ilog2ceil(n);
            int fit_size = pow(2, n_ilog2);
            int fitSizeInBytes = fit_size * sizeof(int);
            dim3 scanFullBlocksPerGrid((fit_size + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_data_array, fitSizeInBytes);
            checkCUDAError("cudaMalloc dev_data_array failed!");

            timer().startGpuTimer();
            // TODO
            // Transform the existing array into zero and one:
            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize>>> (n, dev_zero_one_temp_array, dev_idata_array);

            // Scan:
            // Add zeros at the end of array.
            init_array <<<scanFullBlocksPerGrid, blockSize>>> (dev_data_array, dev_zero_one_temp_array, n, fit_size);

            int d_max = ilog2ceil(n) - 1;
            // Up-Sweep:
            for (int d = 0; d <= d_max; ++d) {
                int threads_num_needed = fit_size * pow(0.5, d + 1);
                dim3 up_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                up_sweep <<<up_sweep_blocks_per_grid, blockSize>>> (dev_data_array, fit_size, d);
            }
            // x[n - 1] = 0
            // int* a = new int[1]();
            // a[0] = 0;
            // cudaMemcpy(dev_data_array + fit_size - 1, a, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_data_array + fit_size - 1, 0, sizeof(int));
            // Down-Sweep:
            for (int d = d_max; d >= 0; --d) {
                int threads_num_needed = fit_size * pow(0.5, d + 1);
                dim3 down_sweep_blocks_per_grid((threads_num_needed + blockSize - 1) / blockSize);
                down_sweep <<<down_sweep_blocks_per_grid, blockSize>>> (dev_data_array, fit_size, d);
            }

            // Get the total number of the final elements:
            int final_count = 0;
            cudaMemcpy(&final_count, dev_data_array + fit_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            // Scatter:
            cudaMalloc((void**)&dev_final_array, final_count * sizeof(int));
            checkCUDAError("cudaMalloc dev_final_array failed!");
            StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (fit_size, dev_final_array, dev_idata_array, dev_zero_one_temp_array, dev_data_array);
            cudaMemcpy(odata, dev_final_array, final_count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            
            timer().endGpuTimer();
            cudaFree(dev_data_array);
            cudaFree(dev_zero_one_temp_array);
            cudaFree(dev_idata_array);
            cudaFree(dev_final_array);
            return final_count;
        }
    }
}
