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

         // separated out so can call in general scan function
         // and in compact with scan function without having
         // an overlapping timer
        void runScan(int n, const int *idata, int *odata) {
          int current_sum = 0;
          for (int i = 0; i < n; ++i) {
            odata[i] = current_sum;
            current_sum += idata[i];
          }
        }

        void scan(int n, int *odata, const int *idata) {
          //-----START
	        timer().startCpuTimer();
          runScan(n, idata, odata);
	        timer().endCpuTimer();
          //-----END
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
          //-----START
	        timer().startCpuTimer();
          
          int current_index = 0;
          for (int i = 0; i < n; ++i) {
            if (idata[i] != 0) {
              odata[current_index] = idata[i];
              ++current_index;
            }
          }

	        timer().endCpuTimer();
          //-----END

          return current_index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        //-----START
          timer().startCpuTimer();

          if (n < 1) {
            return 0;
          }

          // map
          int* map = new int[n];
          int output_length = 0;
          for (int i = 0; i < n; ++i) {
            map[i] = (idata[i] == 0) ? 0 : 1;
          }

          int* scan_output = new int[n];
          runScan(n, map, scan_output);

          // scatter
          int on_output_index = 0;
          for (int i = 1; i < n; ++i) {
            if (scan_output[i] != scan_output[i - 1]) {
              odata[on_output_index] = idata[i - 1];
              ++on_output_index;
            }
          }
          
	        timer().endCpuTimer();
          //-----END

          delete[] map;
          delete[] scan_output;

          return on_output_index;
        }
    }
}
