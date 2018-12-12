
#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define N (8)


inline __device__ half2 greaterThanZero(half2 in) {
  return half2(static_cast<__half2_raw>(in).data.x > 0.0f16, static_cast<__half2_raw>(in).data.y > 0.0f16);
}


__global__ void my_gpu_kernel(hipLaunchParm lp, half2* out0, half2* in0) {
  
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  out0[index] = greaterThanZero(in0[index]);
}


int main() {

  auto memsize = N * sizeof(half);
  
  // initialize the input data
  half* in0 = (half*) malloc (memsize);
    
  for (int i = 0; i < N; i++) {
    in0[i] = -4.0f16 + i;
  }
    
  // allocate the memory on the device side
  half* in0_GPU = nullptr;
  hipMalloc((void**)&in0_GPU, memsize);

  half* out0_GPU = nullptr;
  hipMalloc((void**)&out0_GPU, memsize);

  // allocate memory on the host for the result
  half* out0 = (half*)malloc(memsize);

  // Memory transfer from host to device
  hipMemcpy(in0_GPU, in0, memsize, hipMemcpyHostToDevice);

  // Lauching kernel from host
  hipLaunchKernel(my_gpu_kernel, dim3(1), dim3(4), 0, 0,
		  reinterpret_cast<half2*>(out0_GPU),
		  reinterpret_cast<half2*>(in0_GPU));

  // Memory transfer from device to host
  hipMemcpy(out0, out0_GPU, memsize, hipMemcpyDeviceToHost);

  // print the results
  for (int i = 0; i < N; i++) {
    float f_in0 = in0[i];
    float f_out0 = out0[i];
    std::cout << "( "<< f_in0 << " > 0.0) : " << f_out0 << std::endl;
  }

  // free the resources on device side
  hipFree(in0_GPU);
  hipFree(out0_GPU);

  // free the resources on host side
  free(in0);
  free(out0);

  return 0;
}
