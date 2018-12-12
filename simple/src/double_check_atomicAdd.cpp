#include <atomic>
#include <stdio.h>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM_THREADS (1)
#define DATA_LEN (4)

#define CHECK_DOUBLE

#ifdef CHECK_DOUBLE
using var_dtype_t = double;
#else
using var_dtype_t = float;
#endif

using index_dtype_t = int;


__device__ inline double atomicAddWrapper(double* address, double val) {

  unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull, assumed;
  double* assumed_as_double = reinterpret_cast<double*>(&assumed);

  do {
    assumed = old;
    
    // old = atomicCAS(address_as_ull, assumed,  __double_as_longlong(val + __longlong_as_double(assumed)));

    old = atomicCAS(address_as_ull, assumed,  __double_as_longlong(val + *assumed_as_double));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ inline float atomicAddWrapper(float* address, float val) {

  return atomicAdd(address, val);
}


__global__ void gpuKernel(hipLaunchParm lp, var_dtype_t* params, var_dtype_t update, index_dtype_t param_offset) {
    // int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // int idx = x + y;
    atomicAddWrapper(params + param_offset, update);
}

static void print_var_data(var_dtype_t* data) {
  for (int i=0; i<DATA_LEN; i++) {
    printf("data[%d] = %f\n", i, data[i]);
  }
}


int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    auto var_data_size = DATA_LEN * NUM_THREADS * sizeof(var_dtype_t);
    var_dtype_t* var_data = (var_dtype_t*) malloc (var_data_size);
    for (int i=0; i<DATA_LEN; i++) {
      var_dtype_t data = i;
      var_data[i] = data;
    }

    printf("----- Before -----\n");
    print_var_data(var_data);
    
    // allocate the memory on the device side
    var_dtype_t* var_data_GPU = nullptr;
    hipMalloc((void**)&var_data_GPU, var_data_size);

    // Memory transfer from host to device
    hipMemcpy(var_data_GPU, var_data, var_data_size, hipMemcpyHostToDevice);

    index_dtype_t update_index = 2;
    var_dtype_t update_value = -1;
    
    // Lauching kernel from host
    hipLaunchKernel(gpuKernel, dim3(1), dim3(NUM_THREADS), 0, 0, var_data_GPU, update_value, update_index);

    // Memory transfer from device to host
    hipMemcpy(var_data, var_data_GPU, var_data_size, hipMemcpyDeviceToHost);

    // free the resources on device side
    hipFree(var_data_GPU);

    printf("----- After -----\n");
    print_var_data(var_data);
    
    // free the resources on host side
    free(var_data);

    return 0;
}

