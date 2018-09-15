#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

          int current_sum = 0;
          for (int i = 0; i < n; ++i) {
            odata[i] = current_sum;
            current_sum += idata[i];
          }

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          
          int current_index = 0;
          for (int i = 0; i < n; ++i) {
            if (idata[i] != 0) {
              odata[current_index] = idata[i];
              ++current_index;
            }
          }

	        timer().endCpuTimer();
          return current_index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

          if (n < 1) {
            return 0;
          }

          // map
          int* map = new int[n];
          int output_length = 0;
          for (int i = 1; i < n; ++i) {
            if (idata[i] == 0) {
              map[i] = 0;
            } else {
              map[i] = 1;
              ++output_length;
            }
            
          }

          // scan
          int* scan_output = new int[n];
          int current_sum = 0;
          for (int i = 0; i < n; ++i) {
            scan_output[i] = current_sum;
            current_sum += map[i];
          }

          // scatter
          int prev_in_scan = scan_output[0];
          int on_output_index = 0;
          for (int i = 1; i < n; ++i) {
            if (scan_output[i] != prev_in_scan) {
              odata[on_output_index];
              ++on_output_index;
            }
          }
          
	        timer().endCpuTimer();
          return on_output_index;
        }
    }
}
