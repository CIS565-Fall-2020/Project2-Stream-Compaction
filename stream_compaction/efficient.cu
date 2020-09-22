#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define GLM_FORCE_CUDA
#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //For scan
        int* dev; 

        //For Stream compaction 
        int* dev_SC; 
        int* dev_tempArray; 
        int* dev_scanResult;
        int* dev_finalData;


        //-----------------------------------------------------------------//
        //-----------------------SCAN HELPERS -----------------------------//
        //-----------------------------------------------------------------//

        __global__ void kernFillExtraZeros(int POT, int n, int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;
            if (index > n && index < POT)
            {
                idata[index] = 0;
            }
        }
        
        __global__ void kernUpsweep(int d, int offset, int n, int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;

            if (index < d)
            {
                int a = offset * (2 * index + 1) - 1;
                int b = offset * (2 * index + 2) - 1;
                idata[b] += idata[a];
            }

        }
        
        __global__ void kernClearLastElem(int n, int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;

            if (index == 0)
            {
                idata[n - 1] = 0;
            }
        }
        
        __global__ void kernDownsweep(int d, int offset, int n, int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;
            if (index < d)
            {
                int a = offset * (2 * index + 1) - 1;
                int b = offset * (2 * index + 2) - 1;
                float t = idata[a]; 
                idata[a] = idata[b]; 
                idata[b] += t;
            }

        }
        
        //--------------------------------------------------------//
        //-----------------------SCAN ----------------------------//
        //--------------------------------------------------------//

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {            
            // TODO            
            int POT = 0; 
            int N = 0;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            //If n is not a power of 2, get closest power of 2 and fill the rest of the array with zeroes.
            if (((1 << ilog2ceil(n)) - n) > 0)
            {
                POT = 1 << ilog2ceil(n);
                cudaMalloc((void**)&dev, POT * sizeof(int)); 
                checkCUDAErrorFn("cudaMalloc dev failed!");

                cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAErrorFn("cudaMemcpy dev failed!");

                kernFillExtraZeros << <fullBlocksPerGrid, blockSize >> > (POT, n, dev); 
                N = POT;
            }
            //Else just allocate the device buffer on the gpu 
            else
            {
                cudaMalloc((void**)&dev, n * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev failed!");

                cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAErrorFn("cudaMemcpy dev failed!");
                N = n; 
            }

            timer().startGpuTimer();
           
            //Upsweep 
            int offset = 1;
            for (int d = N >> 1; d > 0; d >>= 1)
            {
                kernUpsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, dev);
                checkCUDAErrorFn("kernUpsweep failed!");
                offset *= 2;
            }

            //Clear Last Element 
            kernClearLastElem << <fullBlocksPerGrid, blockSize >> > (N, dev); 
            checkCUDAErrorFn("kernClearLastElem  failed!");

            //DownSweep 
            //traverse down tree & build scan
            for (int d = 1; d < N; d *= 2)
            {
                offset >>= 1;
                kernDownsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, dev);
                checkCUDAErrorFn("kernDownsweep  failed!");
            }
            timer().endGpuTimer();

            //Write back values to host from device 
            cudaMemcpy(odata, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev to odata failed!");

            cudaFree(dev); 
            
        }

        __global__ void kernComputeTempArray(int n, int* idata, int* tempArray)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
                return;

            if (idata[index] != 0)
            {
                tempArray[index] = 1; 
            }
            else
            {
                tempArray[index] = 0;
            }
        }

        void scanHelper(int n, int* odata, const int* idata) {
            // TODO            
            int POT = 0;
            int N = 0;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            //If n is not a power of 2, get closest power of 2 and fill the rest of the array with zeroes.
            if (((1 << ilog2ceil(n)) - n) > 0)
            {
                POT = 1 << ilog2ceil(n);
                cudaMalloc((void**)&dev, POT * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev failed!");

                cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAErrorFn("cudaMemcpy dev failed!");

                kernFillExtraZeros << <fullBlocksPerGrid, blockSize >> > (POT, n, dev);
                N = POT;
            }
            //Else just allocate the device buffer on the gpu 
            else
            {
                cudaMalloc((void**)&dev, n * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev failed!");

                cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAErrorFn("cudaMemcpy dev failed!");
                N = n;
            }

            //Upsweep 
            int offset = 1;
            for (int d = N >> 1; d > 0; d >>= 1)
            {
                kernUpsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, dev);
                checkCUDAErrorFn("kernUpsweep failed!");
                offset *= 2;
            }

            //Clear Last Element 
            kernClearLastElem << <fullBlocksPerGrid, blockSize >> > (N, dev);
            checkCUDAErrorFn("kernClearLastElem  failed!");

            //DownSweep 
            //traverse down tree & build scan
            for (int d = 1; d < N; d *= 2)
            {
                offset >>= 1;
                kernDownsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, dev);
                checkCUDAErrorFn("kernDownsweep  failed!");
            }

            //Write back values to host from device 
            cudaMemcpy(odata, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev to odata failed!");

            cudaFree(dev);
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

            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_SC, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_SC failed!");

            cudaMalloc((void**)&dev_tempArray, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_tempArray failed!");

            cudaMalloc((void**)&dev_scanResult, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scanResult failed!");

            cudaMalloc((void**)&dev_finalData, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_finalData failed!");

            //Will these be needed? 
            int* tempArray = new int[n];
            int* scanResult = new int[n];

            cudaMemcpy(dev_SC, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy idata to dev_SC failed!");

            timer().startGpuTimer();
            
            //Compute a temp array with 1s and 0s 
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_tempArray, dev_SC);
            checkCUDAErrorFn("kernComputeTempArray failed!");

            //Copy the tempArray to the host Temparray 
            cudaMemcpy(tempArray, dev_tempArray, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy tempArray to dev_tempArray failed!");

            //Scan 
            scanHelper(n, scanResult, tempArray); 

            //Copy the scan result into the device scanResult array 
            cudaMemcpy(dev_scanResult, scanResult, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy scanResult to dev_scanResult failed!");

            //Scatter 
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_finalData, dev_SC, dev_tempArray, dev_scanResult); 
            checkCUDAErrorFn("kernScatter scanResult to dev_scanResult failed!");

            timer().endGpuTimer();

            int count = 0;
            if (idata[n - 1] != 0) 
            {
                count = scanResult[n - 1] + 1;
            }
            else
            {
                count = scanResult[n - 1];
            }

            //Copy data back from device to host 
            cudaMemcpy(odata, dev_finalData, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev_finalData to odata failed!");

            printf("Work Efficient SC count is %d \n", count);
            //printf("Work Efficient Steam Compaction Output is [");
            //for (int i = 0; i < 15; i++)
            //{
            //    if (odata[i])
            //    {
            //        printf(" %d ,", odata[i]);
            //    }
            //}
            //printf("] \n");

            //Delete the heap allocated host memory and free the cuda device memory
            delete[] tempArray;
            delete[] scanResult;
            cudaFree(dev_SC); 
            cudaFree(dev_tempArray);
            cudaFree(dev_scanResult);
            cudaFree(dev_finalData);

            return count;
        }

        //-----------------------------------------------------------------------------//
        //-----------------------S-H-A-R-E-D--M-E-M-O-R-Y------------------------------//
        //-----------------------------------------------------------------------------//


        //__device__ void dev_kernFillExtraZeros(int POT, int n, int* idata)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;
        //    if (index > n && index < POT)
        //    {
        //        idata[index] = 0;
        //    }
        //}

        //__device__ void dev_kernUpsweep(int d, int offset, int n, int* idata)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;

        //    if (index < d)
        //    {
        //        int a = offset * (2 * index + 1) - 1;
        //        int b = offset * (2 * index + 2) - 1;
        //        idata[b] += idata[a];
        //    }

        //}

        //__device__ void dev_kernClearLastElem(int n, int* idata)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;

        //    if (index == 0)
        //    {
        //        idata[n - 1] = 0;
        //    }
        //}

        //__device__ void dev_kernDownsweep(int d, int offset, int n, int* idata)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;
        //    if (index < d)
        //    {
        //        int a = offset * (2 * index + 1) - 1;
        //        int b = offset * (2 * index + 2) - 1;
        //        float t = idata[a];
        //        idata[a] = idata[b];
        //        idata[b] += t;
        //    }

        //}

        //__device__ void dev_kernWriteToSharedMemory(int n, int* shMem, const int* ip)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;
        //    shMem[index] = ip[index];
        //}

        //__device__ void dev_kernWriteFromSharedMemory(int n, int* shMem, int* op)
        //{
        //    int index = threadIdx.x + blockIdx.x * blockDim.x;
        //    if (index >= n)
        //        return;
        //    op[index] = shMem[index];
        //}

        //__global__ void kernScanHelperSharedMemory(int n, int* odata, const int* idata) 
        //{          
        //    int POT = 0;
        //    int N = 0;
        //    dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
        //    extern __shared__ int temp[];

        //    //Write to Shared Memory 
        //    dev_kernWriteToSharedMemory << <fullBlocksPerGrid, blockSize >> > (n, temp, idata);

        //    //If n is not a power of 2, get closest power of 2 and fill the rest of the array with zeroes.
        //    if (((1 << ilog2ceil(n)) - n) > 0)
        //    {
        //        POT = 1 << ilog2ceil(n);
        //        dev_kernFillExtraZeros << <fullBlocksPerGrid, blockSize >> > (POT, n, temp);
        //        N = POT;
        //    }
        //    else
        //    {
        //        N = n;
        //    }

        //    //Upsweep 
        //    int offset = 1;
        //    for (int d = N >> 1; d > 0; d >>= 1)
        //    {
        //        __syncthreads();
        //        dev_kernUpsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, temp);
        //        checkCUDAErrorFn("kernUpsweep failed!");
        //        offset *= 2;
        //    }

        //    //Clear Last Element 
        //    dev_kernClearLastElem << <fullBlocksPerGrid, blockSize >> > (N, temp);
        //    checkCUDAErrorFn("kernClearLastElem  failed!");

        //    //DownSweep 
        //    //traverse down tree & build scan
        //    for (int d = 1; d < N; d *= 2)
        //    {
        //        offset >>= 1;
        //        __syncthreads();
        //        dev_kernDownsweep << <fullBlocksPerGrid, blockSize >> > (d, offset, N, temp);
        //        checkCUDAErrorFn("kernDownsweep  failed!");
        //    }
        //    __syncthreads();

        //    //Write back values to host from device 
        //    dev_kernWriteFromSharedMemory << <fullBlocksPerGrid, blockSize >> > (n, temp, odata);
        //}

        ////HOST FUNCTION 
        ////Work efficient Stream Compaction using shared memory 
        //int compactSharedMemory(int n, int* odata, const int* idata)
        //{
        //    dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

        //    cudaMalloc((void**)&dev_SC, n * sizeof(int));
        //    checkCUDAErrorFn("cudaMalloc dev_SC failed!");

        //    cudaMalloc((void**)&dev_tempArray, n * sizeof(int));
        //    checkCUDAErrorFn("cudaMalloc dev_tempArray failed!");

        //    cudaMalloc((void**)&dev_scanResult, n * sizeof(int));
        //    checkCUDAErrorFn("cudaMalloc dev_scanResult failed!");

        //    cudaMalloc((void**)&dev_finalData, n * sizeof(int));
        //    checkCUDAErrorFn("cudaMalloc dev_finalData failed!");

        //    //Will these be needed? 
        //    int* tempArray = new int[n];
        //    int* scanResult = new int[n];

        //    cudaMemcpy(dev_SC, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        //    checkCUDAErrorFn("cudaMemcpy idata to dev_SC failed!");

        //    timer().startGpuTimer();

        //    //Compute a temp array with 1s and 0s 
        //    Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_tempArray, dev_SC);
        //    checkCUDAErrorFn("kernComputeTempArray failed!");

        //    //Copy the tempArray to the host Temparray 
        //    cudaMemcpy(tempArray, dev_tempArray, n * sizeof(int), cudaMemcpyDeviceToHost);
        //    checkCUDAErrorFn("cudaMemcpy tempArray to dev_tempArray failed!");

        //    //Scan 
        //    //Calling a device function with __global__ signature 
        //    kernScanHelperSharedMemory << <fullBlocksPerGrid, blockSize >> > (n, scanResult, tempArray); 

        //    //Copy the scan result into the device scanResult array 
        //    cudaMemcpy(dev_scanResult, scanResult, n * sizeof(int), cudaMemcpyHostToDevice);
        //    checkCUDAErrorFn("cudaMemcpy scanResult to dev_scanResult failed!");

        //    //Scatter 
        //    Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_finalData, dev_SC, dev_tempArray, dev_scanResult);
        //    checkCUDAErrorFn("kernScatter scanResult to dev_scanResult failed!");

        //    timer().endGpuTimer();

        //    int count = 0;
        //    if (idata[n - 1] != 0)
        //    {
        //        count = scanResult[n - 1] + 1;
        //    }
        //    else
        //    {
        //        count = scanResult[n - 1];
        //    }

        //    //Copy data back from device to host 
        //    cudaMemcpy(odata, dev_finalData, count * sizeof(int), cudaMemcpyDeviceToHost);
        //    checkCUDAErrorFn("cudaMemcpy dev_finalData to odata failed!");

        //    printf("Work Efficient SC count is %d \n", count);

        //    //Delete the heap allocated host memory and free the cuda device memory
        //    delete[] tempArray;
        //    delete[] scanResult;
        //    cudaFree(dev_SC);
        //    cudaFree(dev_tempArray);
        //    cudaFree(dev_scanResult);
        //    cudaFree(dev_finalData);

        //    return count;
        //}

    }
}

