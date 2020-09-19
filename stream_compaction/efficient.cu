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

        __global__ void upSweep(int *data, int d) {
          int idx = threadIdx.x + (blockIdx.x * blockDim.x);
          int interval = 1 << d;
          int mapped = interval * idx + interval - 1;
          data[mapped] += data[mapped - (interval >> 1)];
        }

        __global__ void downSweep(int *data, int d) {
          int idx = threadIdx.x + (blockIdx.x * blockDim.x);
          int interval = 1 << d;
          int node = interval * idx + interval - 1;
          int left = node - (interval >> 1);
          int temp = data[left];
          data[left] = data[node];
          data[node] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *dev_odata, const int *dev_idata, bool callFromMain) {
          int iterations = ilog2ceil(n);
          int nextN = 1 << iterations;
          int *dev_idata_temp;
          cudaMalloc((void **) &dev_idata_temp, nextN * sizeof(int));
          checkCUDAError("cudaMalloc dev_idata_temp failed");
          cudaMemset(dev_idata_temp, 0, nextN *sizeof(int));
          checkCUDAError("cudaMemset dev_idata_temp failed");
          if (callFromMain) {
              cudaMemcpy(dev_idata_temp, dev_idata, sizeof(int) * n, cudaMemcpyHostToDevice);
              timer().startGpuTimer();
          }
          else {
              cudaMemcpy(dev_idata_temp, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
          }
          checkCUDAError("cudaMemcpy dev_idata_temp failed");

          // Up-sweep
          for (int d = 1; d <= iterations; d++) {
            int numThreads = 1 << (iterations - d);
            dim3 blocks((numThreads + blockSize - 1) / blockSize);
            upSweep<<<blocks, blockSize>>>(dev_idata_temp, d);
            checkCUDAError("upSweep failed");
          }

          // Down-sweep
          // Set the "root" to 0
          cudaMemset(&dev_idata_temp[nextN - 1], 0, sizeof(int));
          for (int d = iterations; d >= 1; d--) {
            int numThreads = 1 << (iterations - d);
            dim3 blocks((numThreads + blockSize - 1) / blockSize);
            downSweep<<<blocks, blockSize>>>(dev_idata_temp, d);
            checkCUDAError("downSweep failed");
          }
          
          if (callFromMain) {
              timer().endGpuTimer();
              cudaMemcpy(dev_odata, dev_idata_temp, sizeof(int) * n, cudaMemcpyDeviceToHost);
          }
          else {
              cudaMemcpy(dev_odata, dev_idata_temp, sizeof(int) * n, cudaMemcpyDeviceToDevice);
          }
          checkCUDAError("cudaMemcpy dev_odata failed");

          cudaFree(dev_idata_temp);
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
          int *bools, *indices, *dev_idata, *dev_odata;
          cudaMalloc((void**) &bools, sizeof(int) * n);
          cudaMalloc((void**) &indices, sizeof(int) * n);
          cudaMalloc((void**) &dev_idata, sizeof(int) * n);
          cudaMalloc((void**) &dev_odata, sizeof(int) * n);
          cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

          timer().startGpuTimer();

          dim3 blocks((n + blockSize - 1) / blockSize);
          Common::kernMapToBoolean<<<blocks, blockSize>>>(n, bools, dev_idata);
          scan(n, indices, bools, false);
          Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, bools, indices);

          timer().endGpuTimer();
          
          int cnt, lastBool;
          cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
          // Copy the count back
          cudaMemcpy(&cnt, &indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(&lastBool, &bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(bools);
          cudaFree(indices);
          cudaFree(dev_idata);
          cudaFree(dev_odata);
          return lastBool ? cnt + 1 : cnt;
        }
    }
}
