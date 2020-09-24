#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        int* dev_tempData;
        int* dev_inputData;
        int* dev_boolData;
        int* dev_notBoolData;
        int* dev_scanData;
        int* dev_outputData;

        // Returns the position of the most significant bit
        int getMSB(int x)
        {
            int bit = 1 << 31;
            for (int i = 31; i >= 0; i--, bit >>= 1)
            {
                if (x & bit)
                    return i + 1;
            }
            return 0;
        }

        // Returns the maximum of the array
        int getMax(int n, const int* a)
        {
            int maximum = a[0];
            for (int i = 1; i < n; i++)
            {
                maximum = std::max(maximum, a[i]);
            }
            return maximum;
        }

        // Maps an array to 2 arrays only contains 1s and 0s. 
        // _bools_ is just the logic NOT of _notBools_
        __global__ void kernMapTo2Bools(int n, int bit, int* bools, int* notBools, const int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            bool b = idata[index] & bit;
            bools[index] = b;
            notBools[index] = !b;
        }

        // Computes the temp array _temps_ which stores address for writing true keys
        __global__ void kernComputeAddressOfTrueKeys(int n, int* temps, const int* notBools, const int* scanData)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
           
            int totalFalses = notBools[n - 1] + scanData[n - 1];
            temps[index] = index - scanData[index] + totalFalses;
        }

        // Scatters based on address _temps_
        __global__ void kernRadixScatter(int n, int* odata, const int* temps, const int* bools, const int* scanData, const int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            int newIdx = bools[index] ? temps[index] : scanData[index];
            odata[newIdx] = idata[index];
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void sort(int n, int* odata, const int* idata)
        {
            int depth = ilog2ceil(n);
            int size = 1 << depth;  // sizes of arrays will are rounded to the next power of two
            int maximum = getMax(n, idata);
            int highestBit = getMSB(maximum);

            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 scanBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_inputData, n * sizeof(int));
            cudaMalloc((void**)&dev_boolData, n * sizeof(int));
            cudaMalloc((void**)&dev_notBoolData, n * sizeof(int));
            cudaMalloc((void**)&dev_scanData, size * sizeof(int));
            cudaMalloc((void**)&dev_tempData, n * sizeof(int));
            cudaMalloc((void**)&dev_outputData, n * sizeof(int));
            Common::kernInitializeArray<<<scanBlocksPerGrid, threadsPerBlock>>>(size, dev_scanData, 0);
            cudaMemcpy(dev_inputData, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            // Do radix sort for _bits_ times
            for (int i = 0, bit = 1; i < highestBit; i++, bit <<= 1)
            {
                // Step 1: Compute the bool array and notBool array
                kernMapTo2Bools<<<blocksPerGrid, threadsPerBlock>>>(n, bit, dev_boolData, dev_notBoolData, dev_inputData);

                // Step 2: Exclusive scan array
                cudaMemcpy(dev_scanData, dev_notBoolData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                for (int d = 0; d < depth; d++)
                {
                    Efficient::kernUpSweep<<<scanBlocksPerGrid, threadsPerBlock>>>(size, dev_scanData, d);
                }

                cudaMemset(dev_scanData + size - 1, 0, sizeof(int));
                for (int d = depth - 1; d >= 0; d--)
                {
                    Efficient::kernDownSweep<<<scanBlocksPerGrid, threadsPerBlock>>>(size, dev_scanData, d);
                }

                // Step 3: Compute temp array _dev_tempData_
                kernComputeAddressOfTrueKeys<<<blocksPerGrid, threadsPerBlock>>>(n, dev_tempData, dev_notBoolData, dev_scanData);

                // Step 4: Scatter
                kernRadixScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dev_outputData, dev_tempData, dev_boolData, dev_scanData, dev_inputData);

                // Swap for next round of radix sort
                std::swap(dev_outputData, dev_inputData);
            }

            cudaMemcpy(odata, dev_inputData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(dev_inputData);
            cudaFree(dev_boolData);
            cudaFree(dev_notBoolData);
            cudaFree(dev_scanData);
            cudaFree(dev_tempData);
            cudaFree(dev_outputData);
        }
    }
}
