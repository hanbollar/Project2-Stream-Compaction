#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelEfficientScanUpSweep(int array_length, int offset, int *data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= array_length) {
            return;
          }

          int next_offset = offset * 2;

          // we only want to sum with threads that affect our next iteration
          if (index % next_offset == 0 && index + next_offset <= array_length) {
            data[index + next_offset - 1] += data[index + offset - 1];
          }
        }

        __global__ void kernelEfficientScanDownSweep(int array_length, int offset, int *data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= array_length) {
            return;
          }

          int next_offset = 2 * offset;

          // we only want to work with threads that affect our next iteration
          if (index % next_offset == 0 && index + next_offset <= array_length) {
            int a_index = index + offset - 1;
            int b_index = index + next_offset - 1;

            int temp = data[a_index];
            data[a_index] = data[b_index];
            data[b_index] += temp;
          }
        }

        /**
        * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        */

        // writing this here so that i can call it in compact and 
        // avoid the timer conflict issue
        void runScan(int n, int *odata, const int *idata) {
          int max_passes = ilog2ceil(n);
          int upper_bound = 1 << max_passes;

          dim3 blocksPerGrid((upper_bound + blockSize - 1) / blockSize);
          dim3 threadsPerBlock(blockSize);

          // array used for in-place threaded manipulations
          int *dev_temp;
          cudaMalloc((void**)&dev_temp, sizeof(int) * upper_bound);
          // zero out array so also zeroing the unneeded elements past length n --> upper_bound
          cudaMemset(dev_temp, 0, sizeof(int) * upper_bound);

          cudaMemcpy(dev_temp, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

          // BEGIN: efficient scan upsweep
          for (int on_pass = 1; on_pass < upper_bound; on_pass *= 2) {
            kernelEfficientScanUpSweep << <blocksPerGrid, threadsPerBlock >> > (upper_bound, on_pass, dev_temp);
            checkCUDAError("kernelEfficientScanUpSweep failed!", __LINE__);
          }
          // END: efficient scan upsweep

          // BEGIN: efficient scan downsweep
          // set max element from upsweep iteration, dev_temp[n - 1], to 0
          cudaMemset(dev_temp + upper_bound - 1, 0, sizeof(int));
          checkCUDAError("copying zero failed!", __LINE__);

          for (int pass_iteration = upper_bound / 2; pass_iteration > 0; pass_iteration /= 2) {
            kernelEfficientScanDownSweep << <blocksPerGrid, threadsPerBlock >> > (upper_bound, pass_iteration, dev_temp);
            checkCUDAError("kernelEfficientScanUpSweep failed!", __LINE__);
          }
          // END: efficient scan downsweep

          cudaMemcpy(odata, dev_temp, sizeof(int)*n, cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy dev_swapA and dev_swapB failed!", __LINE__);

          cudaFree(dev_temp);
        }

        void scan(int n, int *odata, const int *idata) {
          //-----START
          timer().startGpuTimer();
          runScan(n, odata, idata);
          timer().endGpuTimer();
          //-----END
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
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            int* data;
            int* final_data;
            int* bools;
            int* scan;
            cudaMalloc((void**)&data, sizeof(int) * n);
            cudaMalloc((void**)&final_data, sizeof(int) * n);
            cudaMalloc((void**)&bools, sizeof(int) * n);
            cudaMalloc((void**)&scan, sizeof(int) * n);
            checkCUDAError("mallocing failed!", __LINE__);

            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("copy to gpu data array failed!", __LINE__);

            //-----START
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlock>> > (n, bools, data);
            checkCUDAError("kernMapToBoolean failed!", __LINE__);

            StreamCompaction::Efficient::runScan(n, scan, bools);

            int last_value[1];
            cudaMemcpy(&last_value, scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = (idata[n - 1] == 0) ? last_value[0] : last_value[0] + 1;

            StreamCompaction::Common::kernScatter << <blocksPerGrid, threadsPerBlock>> > (n, final_data, data, bools, scan);
            checkCUDAError("kernScatter failed!", __LINE__);
            timer().endGpuTimer();
            //-----END

            cudaMemcpy(odata, final_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("copy from gpu data array failed!", __LINE__);

            cudaFree(data);
            cudaFree(final_data);
            cudaFree(bools);
            cudaFree(scan);

            return count;
        }

      /*
      * Same idea as StreamCompaction::Radix except instead returns the opposite
      * boolean as the other function.
      */
      __global__ void kernMapToSpecifiedDigit(int n, const int binary_digit, int *opp_binary, const int *idata) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= n) {
          return;
        }

        // we want opp boolean of the binary masking's output
        opp_binary[index] = (idata[index] & binary_digit) ? 0 : 1;
      }

      /*
      * Implementing intermediary steps of gpu gems 3 section 39.3.3 for radix sort
      * using t and d values to create final index outputs to be used in scatter
      * to move all the elements to their final sorted ordering for the current digit
      * iteration
      */
      __global__ void kernResolveScatterInput(int n, int *scatter_input, const int num_false, const int* opp_binary, const int *scan_output) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= n) {
          return;
        }

        // t = i - f + num_false

        // b = !digits since i set digits to be the opposite of the normal binary rep for inputting to scan
        // d = b ? t : f
        //scatter_input[index] = index - scan_output[index] + num_false;

        scatter_input[index] = (!opp_binary[index]) ? index - scan_output[index] + num_false : scan_output[index];
      }

      /*
      * Scatter where all bools are true
      */
      __global__ void kernTrueScatter(int n, int *odata, const int *idata, const int *indices) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= n) {
          return;
        }

        odata[indices[index]] = idata[index];
      }

      /*
      * The CUDA implementation of radix sort on integer inputs
      */
      void radix(int n, int *odata, const int *idata, const int max_digit) {
        int upper_bound = 1 << max_digit;


        dim3 blocksPerGrid((upper_bound + blockSize - 1) / blockSize);
        dim3 threadsPerBlock(blockSize);

        int* dev_opposite_binary;
        int* dev_scan_output;
        int* dev_idata;
        int* dev_scatter_indices_input;
        int* dev_odata;
        cudaMalloc((void**)&dev_opposite_binary, sizeof(int) * upper_bound);
        cudaMalloc((void**)&dev_scan_output, sizeof(int) * upper_bound);
        cudaMalloc((void**)&dev_idata, sizeof(int) * upper_bound);
        cudaMalloc((void**)&dev_scatter_indices_input, sizeof(int) * upper_bound);
        cudaMalloc((void**)&dev_odata, sizeof(int) * upper_bound);
        checkCUDAError("mallocing failed!", __LINE__);

        cudaMemset(dev_idata, 0, sizeof(int) * upper_bound);
        cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

        //START
        timer().startGpuTimer();
        //int i = 1;
        for (int i = 1; i <= max_digit; i *= 2) {
          // map to opposite binary
          kernMapToSpecifiedDigit << <blocksPerGrid, threadsPerBlock >> > (n, i, dev_opposite_binary, dev_idata);

          // scan
          StreamCompaction::Efficient::runScan(n, dev_scan_output, dev_opposite_binary);

          // interpret scan input
          int last_value[1];
          cudaMemcpy(&last_value, dev_scan_output + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
          last_value[0] += (int)(last_value[0] != 0); //(int)(n % 2 == 0);
          kernResolveScatterInput << <blocksPerGrid, threadsPerBlock >> > (n, dev_scatter_indices_input, last_value[0], dev_opposite_binary, dev_scan_output);

          // perform scatter
          kernTrueScatter << <blocksPerGrid, threadsPerBlock >> >(n, dev_odata, dev_idata, dev_scatter_indices_input);
          cudaMemcpy(dev_idata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
        }
        timer().endGpuTimer();
        //-----END

        cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

        cudaFree(dev_opposite_binary);
        cudaFree(dev_scan_output);
        cudaFree(dev_idata);
        cudaFree(dev_scatter_indices_input);
        cudaFree(dev_odata);
      }
    }
}
