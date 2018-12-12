#include <stdio.h>
#include <hip/hip_fp16.h>


#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM (8)

using input_type_t = half;
using output_type_t = unsigned short int;

void print_values(half h, unsigned short int u)
{
  printf ("%f : %x\n", h, u);
}


__global__ void half_2_raw_ushort(hipLaunchParm lp, output_type_t* out, input_type_t* in) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int idx = x + y;
    out[idx] = __half_as_ushort(in[idx]);
}


int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    input_type_t* input = (input_type_t*) malloc (NUM * sizeof(input_type_t));
    input[0] = 0;
    input[1] = 1;
    input[2] = 2;
    input[3] = 4;
    input[4] = 0.5;
    input[5] = 0.25;
    input[6] = 0.125;
    input[7] = 0.0625;
    
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
    hipLaunchKernel(half_2_raw_ushort, dim3(1), dim3(NUM), 0, 0, resultOnGPU, inputOnGPU);

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

