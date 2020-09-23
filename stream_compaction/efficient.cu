#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n_pot, int* data, int d)
        {   
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            int temp_d = 1 << (d + 1);
            int k = index * temp_d;

            if (k >= n_pot) {
                return;
            }

            int power1 = 1 << (d + 1);
            int power2 = 1 << d;
            data[k + power1 - 1] += data[k + power2 - 1];
        }

        __global__ void kernDownSweep(int n_pot, int* data, int d)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            int temp_d = 1 << (d + 1);
            int k = index * temp_d;

            if (k >= n_pot) {
                return;
            }      
            
            int power1 = 1 << (d + 1);
            int power2 = 1 << d;
            int t = data[k + power2 - 1];
            data[k + power2 - 1] = data[k + power1 - 1];
            data[k + power1 - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_data;

            // Get power of two length
            int logValue = ilog2ceil(n);
            int n_pot =  1 << logValue;

            // CUDA memory arrangement and error checking
            cudaMalloc((void**)&dev_data, n_pot * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemset(dev_data, 0, n_pot * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_data failed!");

            timer().startGpuTimer();
            
            // Up-Sweep
            for(int d = 0; d <= ilog2ceil(n) - 1; ++d)
            {
                dim3 blocksPerGrid((n_pot / pow(2, d + 1) + blockSize - 1) / blockSize);
                kernUpSweep<<<blocksPerGrid, blockSize>>>(n_pot, dev_data, d);
            }

            // Down-Sweep
            cudaMemset(dev_data + n_pot - 1, 0, sizeof(int)); 
            for(int d = ilog2ceil(n) - 1; d >=0; --d)
            {
                dim3 blocksPerGrid((n_pot / pow(2, d + 1) + blockSize - 1) / blockSize);
                kernDownSweep<<<blocksPerGrid, blockSize>>>(n_pot, dev_data, d);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_data to idata failed!");

            cudaFree(dev_data);
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
            
            int* dev_idata;
            int* dev_boolData;
            int* dev_indices;
            int* dev_odata;


            int logValue = ilog2ceil(n);
            int n_pot = 1 << logValue;

            cudaMalloc((void**)&dev_idata, n_pot * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_boolData, n_pot * sizeof(int));
            checkCUDAError("cudaMalloc dev_boolData failed!");

            cudaMalloc((void**)&dev_indices, n_pot * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMemset(dev_idata, 0, n_pot * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_idata failed!");

            timer().startGpuTimer();
            
            dim3 mapBlocksPerGrid((n_pot + blockSize - 1) / blockSize);

            // Compute temporary array containing 1 and 0
            Common::kernMapToBoolean<<<mapBlocksPerGrid, blockSize>>>(n_pot, dev_boolData, dev_idata);

            cudaMemcpy(dev_indices, dev_boolData, n_pot * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy from dev_boolData to dev_indices failed!");

            // Run exclusive scan on mapped array
            // Up-Sweep
            for(int d = 0; d <= ilog2ceil(n) - 1; ++d)
            {
                dim3 blocksPerGrid((n_pot / pow(2, d + 1) + blockSize - 1) / blockSize);
                kernUpSweep<<<blocksPerGrid, blockSize>>>(n_pot, dev_indices, d);
            }

            // Down-Sweep
            cudaMemset(dev_indices + n_pot - 1, 0, sizeof(int)); 
            for(int d = ilog2ceil(n) - 1; d >=0; --d)
            {
                dim3 blocksPerGrid((n_pot / pow(2, d + 1) + blockSize - 1) / blockSize);
                kernDownSweep<<<blocksPerGrid, blockSize>>>(n_pot, dev_indices, d);
            }

            // Scatter
            
            int arrayCount = 0;
            cudaMemcpy(&arrayCount, dev_indices + n_pot - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMalloc((void**)&dev_odata, arrayCount * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            Common::kernScatter<<<mapBlocksPerGrid, blockSize>>>(n_pot, dev_odata, dev_idata, dev_boolData, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, arrayCount * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_boolData);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return arrayCount;
        }
    }
}
