#include <stdio.h>
#include <hip/hip_fp16.h>


#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM (8)

using input_type_t = unsigned short int;
using output_type_t = int;

void print_values(input_type_t in, output_type_t out)
{
  printf ("%x : %d\n", in, out);
}


__global__ void gpuKernel(hipLaunchParm lp, output_type_t* out, input_type_t* in) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int idx = x + y;
    out[idx] = __hisinf(__ushort_as_half(in[idx]));
}


int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    input_type_t* input = (input_type_t*) malloc (NUM * sizeof(input_type_t));
    input[0] = 0xfc00;
    input[1] = 0xfc01;
    input[2] = 0xfc10;
    input[3] = 0x9234;
    input[4] = 0x7c00;
    input[5] = 0x7c01;
    input[6] = 0x7c10;
    input[7] = 0x1234;
    
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

