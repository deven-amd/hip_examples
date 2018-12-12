#include <stdio.h>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM_THREADS (1)
#define DATA_LEN (8)

using var_dtype_t = unsigned long long;

using index_dtype_t = int;


__device__ inline var_dtype_t atomicAddWrapper(var_dtype_t* address, var_dtype_t val) {

  var_dtype_t old = *address, assumed;

  do {
    assumed = old;
    old = atomicCAS(address, assumed,  val + assumed);
    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return old;
}



__global__ void gpuKernel(hipLaunchParm lp, var_dtype_t* params, var_dtype_t update, index_dtype_t param_offset) {
    // int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // int idx = x + y;
    atomicAddWrapper(params + param_offset, update);
}

static void print_var_data(var_dtype_t* data) {
  for (int i=0; i<DATA_LEN; i++) {
    printf("data[%d] = %llx\n", i, data[i]);
  }
}


int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    auto var_data_size = DATA_LEN * NUM_THREADS * sizeof(var_dtype_t);
    var_dtype_t* var_data = (var_dtype_t*) malloc (var_data_size);
    // var_data[0] = -1.3447058;
    // var_data[1] = -0.31186214;
    // var_data[2] = 1.271002;
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

    index_dtype_t update_index = 3;
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

