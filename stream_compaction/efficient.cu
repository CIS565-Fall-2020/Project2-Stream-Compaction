#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        int* dev_tempData;
        int* dev_inputData;
        int* dev_boolData;
        int* dev_idxData;
        int* dev_outputData;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* tdata, int d)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            int leftChildOffset = 1 << d;  // 2^d          
            int rightChildOffset = leftChildOffset << 1;  // 2^(d+1)   
            if (index % rightChildOffset == 0)
            {
                tdata[index + rightChildOffset - 1] += tdata[index + leftChildOffset - 1];
            }
        }

        __global__ void kernDownSweep(int n, int* odata, int d)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            int leftChildOffset = 1 << d;  // 2^d          
            int rightChildOffset = leftChildOffset << 1;  // 2^(d+1)   
            if (index % rightChildOffset == 0)
            {
                // Save left child value
                int preLeftChildVal = odata[index + leftChildOffset - 1];
                // Set the left child of the next round as the current node's value
                odata[index + leftChildOffset - 1] = odata[index + rightChildOffset - 1];
                // Set the right child (the node itself) of the next round as the current node's value + previous left child value
                odata[index + rightChildOffset - 1] += preLeftChildVal;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int depth = ilog2ceil(n);
            int size = 1 << depth;  // sizes of arrays will are rounded to the next power of two
            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((size + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_tempData, size * sizeof(int));
            Common::kernInitializeArray<<<blocksPerGrid, threadsPerBlock>>>(size, dev_tempData, 0);
            cudaMemcpy(dev_tempData, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // ------------------------------------- Performance Measurement -------------------------------------------
            timer().startGpuTimer();
            
            for (int d = 0; d < depth; d++)
            {
                kernUpSweep<<<blocksPerGrid, threadsPerBlock>>>(size, dev_tempData, d);
            }
    
            // Set root to zero
            cudaMemset(dev_tempData + size - 1, 0, sizeof(int));
            for (int d = depth - 1; d >= 0; d--)
            {
                kernDownSweep<<<blocksPerGrid, threadsPerBlock>>>(size, dev_tempData, d);
            }

            timer().endGpuTimer();
            // --------------------------------------------------------------------------------------------------------

            cudaMemcpy(odata, dev_tempData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(dev_tempData);
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
        int compact(int n, int *odata, const int *idata) 
        {   
            int depth = ilog2ceil(n);
            int size = 1 << depth;  // sizes of arrays will are rounded to the next power of two
            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((size + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_inputData, size * sizeof(int));
            cudaMalloc((void**)&dev_boolData, size * sizeof(int));
            cudaMalloc((void**)&dev_idxData, size * sizeof(int));
            cudaMalloc((void**)&dev_outputData, n * sizeof(int));
            Common::kernInitializeArray<<<blocksPerGrid, threadsPerBlock>>>(size, dev_inputData, 0);
            Common::kernInitializeArray<<<blocksPerGrid, threadsPerBlock>>>(size, dev_boolData, 0);
            Common::kernInitializeArray<<<blocksPerGrid, threadsPerBlock>>>(size, dev_idxData, 0);
            Common::kernInitializeArray<<<blocksPerGrid, threadsPerBlock>>>(n, dev_outputData, 0);
            cudaMemcpy(dev_inputData, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            
            // ------------------------------------- Performance Measurement -------------------------------------------
            timer().startGpuTimer();
            // Step 1: Compute temporary array
            Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, dev_boolData, dev_inputData);

            // Step 2: Run exclusive scan on temporary array
            cudaMemcpy(dev_idxData, dev_boolData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            for (int d = 0; d < depth; d++)
            {
                kernUpSweep<<<blocksPerGrid, threadsPerBlock>>>(size, dev_idxData, d);
            }

            cudaMemset(dev_idxData + size - 1, 0, sizeof(int));
            for (int d = depth - 1; d >= 0; d--)
            {
                kernDownSweep<<<blocksPerGrid, threadsPerBlock>>>(size, dev_idxData, d);
            }

            // Step 3: Scatter
            Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dev_outputData, dev_inputData, dev_boolData, dev_idxData);

            timer().endGpuTimer();
            // --------------------------------------------------------------------------------------------------------

            int compactedSize = -1;
            cudaMemcpy(&compactedSize, dev_idxData + size - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_outputData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(dev_inputData);
            cudaFree(dev_boolData);
            cudaFree(dev_idxData);
            cudaFree(dev_outputData);
            return compactedSize;
        }
    }
}
