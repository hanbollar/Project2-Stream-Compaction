#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_input;
            int* dev_output;
            cudaMalloc((void**)&dev_input, sizeof(int) * n);
            cudaMalloc((void**)&dev_output, sizeof(int) * n);

            cudaMemcpy(dev_input, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            thrust::device_ptr<int> input(dev_input);
            thrust::device_ptr<int> output(dev_output);

            timer().startGpuTimer();
            thrust::exclusive_scan(input, input + n, output);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_output, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_input);
            cudaFree(dev_output);
        }
    }
}