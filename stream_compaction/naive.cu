#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 32

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernelNaiveScan(int n, int offset, const int *idata, int *odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          
          if (index >= offset) {
            odata[index] = idata[index  - offset] + idata[index];
          } else {
            odata[index] = idata[index];
          }
        }

        __global__ void kernelScanShift(int n, const int *idata, int *odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }

          if (index > 0) {
            odata[index] = idata[index - 1]; 
          } else {
            odata[0] = 0;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // two arrays to swap between while scanning
            int *dev_swapA;
            int *dev_swapB;

            cudaMalloc((void**)&dev_swapA, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_swapA failed!", __LINE__);

            cudaMalloc((void**)&dev_swapB, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_swapB failed!", __LINE__);

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            // -----START
            timer().startGpuTimer();

            // copy over all the values of idata into A
            cudaMemcpy(dev_swapA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into dev_swapA failed!", __LINE__);

            // scan
            for (int pass_iteration = 1; pass_iteration < n; pass_iteration *= 2) {
              kernelNaiveScan << <blocksPerGrid, threadsPerBlock >> > (n, pass_iteration, dev_swapA, dev_swapB);
              checkCUDAError("kernelNaiveScan failed!", __LINE__);
              
              // alternate A and B
              cudaMemcpy(dev_swapA, dev_swapB, sizeof(int)*n, cudaMemcpyDeviceToDevice);
              checkCUDAError("cudaMemcpy dev_swapA and dev_swapB failed!", __LINE__);
            }

            kernelScanShift << <blocksPerGrid, threadsPerBlock >> > (n, dev_swapA, dev_swapB);
            checkCUDAError("kernelScanShift failed!", __LINE__);

            cudaMemcpy(odata, dev_swapB, sizeof(int)*n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_swapA and dev_swapB failed!", __LINE__);

            timer().endGpuTimer();
            // -----END

            cudaFree(dev_swapA);
            cudaFree(dev_swapB);
        }
    }
}
