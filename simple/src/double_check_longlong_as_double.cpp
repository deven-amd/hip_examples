#include <stdio.h>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

//#define IMPLEMENT_WORKAROUND

#define NUM (8)

using input_type_t = long long;
using output_type_t = double;

void print_values(input_type_t in, output_type_t out)
{
  printf ("%016llx : %f\n", in, out);
}


__global__ void gpuKernel(hipLaunchParm lp, output_type_t* out, input_type_t* in) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int idx = x + y;

#ifdef IMPLEMENT_WORKAROUND    

    output_type_t* addr = reinterpret_cast<output_type_t*>(&in[idx]);
    out[idx] =  *addr;

#else

    out[idx] = __longlong_as_double(in[idx]);

#endif
}


int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    input_type_t* input = (input_type_t*) malloc (NUM * sizeof(input_type_t));
    input[0] = 0x0000000000000000ull;  // bit-pattern for +0.0
    input[1] = 0x3ff0000000000000ull;  // bit-pattern for 1.0
    input[2] = 0x4000000000000000ull;  // bit-pattern for 2.0
    input[3] = 0x4008000000000000ull;  // bit-pattern for 3.0
    input[4] = 0x4010000000000000ull;  // bit-pattern for 4.0
    input[5] = 0x4014000000000000ull;  // bit-pattern for 5.0
    input[6] = 0x7ff0000000000000ull;  // bit-pattern for +inf
    input[7] = 0x7ff8000000000000ull;  // bit-pattern for nan
    
    // allocate the memory on the device side
    input_type_t* inputOnGPU = nullptr;
    output_type_t* resultOnGPU = nullptr;
    hipMalloc((void**)&inputOnGPU, NUM * sizeof(input_type_t));
    hipMalloc((void**)&resultOnGPU, NUM * sizeof(output_type_t));

    // allocate memory on the host for the result
    output_type_t* result = (output_type_t*)malloc(NUM * sizeof(output_type_t));

    // Memory transfer from host to device
    hipMemcpy(inputOnGPU, input, NUM * sizeof(input_type_t), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernel(gpuKernel, dim3(1), dim3(NUM), 0, 0, resultOnGPU, inputOnGPU);

    // Memory transfer from device to host
    hipMemcpy(result, resultOnGPU, NUM * sizeof(output_type_t), hipMemcpyDeviceToHost);

    // free the resources on device side
    hipFree(inputOnGPU);
    hipFree(resultOnGPU);

    for (int i=0; i<NUM; i++) {
      print_values(input[i], result[i]);
    }
    
    // free the resources on host side
    free(input);
    free(result);

    return 0;
}

