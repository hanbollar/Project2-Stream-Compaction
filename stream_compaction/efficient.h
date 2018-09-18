#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
      StreamCompaction::Common::PerformanceTimer& timer();

      // writing this here so that i can call it in compact and 
      // avoid the timer conflict issue
      void runScan(int n, int *odata, const int *idata);

      void scan(int n, int *odata, const int *idata);

      int compact(int n, int *odata, const int *idata);

      /*
      * The CUDA implementation of radix sort on integer inputs
      */
      void radix(int n, int *odata, const int *idata, const int max_digit);
    }
}
