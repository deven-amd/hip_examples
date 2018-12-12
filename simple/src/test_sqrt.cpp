
#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM (8)

using datatype_t = half;

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void sqrtGPU(hipLaunchParm lp, datatype_t* out, datatype_t* in) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int idx = x + y;
    out[idx] = hsqrt(in[idx]);
}

// CPU implementation 
void sqrtCPU(datatype_t* output, datatype_t* input) {
  for (unsigned int idx = 0; idx < NUM; idx++) {
    output[idx] = sqrt(input[idx]);
  }
}

int main() {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // initialize the input data
    datatype_t* input = (datatype_t*) malloc (NUM * sizeof(datatype_t));
    for (int i = 0; i < NUM; i++) {
        input[i] = (datatype_t)i;
    }
    input[7] = -65504.0; 
    
    // allocate the memory on the device side
    datatype_t* inputOnGPU = nullptr;
    datatype_t* resultOnGPU = nullptr;
    hipMalloc((void**)&inputOnGPU, NUM * sizeof(datatype_t));
    hipMalloc((void**)&resultOnGPU, NUM * sizeof(datatype_t));

    // allocate memory on the host for the result
    datatype_t* gpuResult = (datatype_t*)malloc(NUM * sizeof(datatype_t));
    datatype_t* cpuResult = (datatype_t*)malloc(NUM * sizeof(datatype_t));

    // Memory transfer from host to device
    hipMemcpy(inputOnGPU, input, NUM * sizeof(datatype_t), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernel(sqrtGPU, dim3(1), dim3(NUM), 0, 0, resultOnGPU, inputOnGPU);

    // Memory transfer from device to host
    hipMemcpy(gpuResult, resultOnGPU, NUM * sizeof(datatype_t), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    sqrtCPU(cpuResult, input);

    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
      if (i <=16) {
	printf("%d : %.6f : %.6f\n", i, gpuResult[i], cpuResult[i]);
      }
      if (std::abs((float)gpuResult[i] - (float)cpuResult[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(inputOnGPU);
    hipFree(resultOnGPU);

    // free the resources on host side
    free(input);
    free(gpuResult);
    free(cpuResult);

    return errors;
}
