#include <iostream>

// hip header file
#include "hip/hip_runtime.h"

typedef unsigned short bfloat16;
typedef unsigned short uint16_t;

inline __host__ __device__ void ToBFloat16(const float* src, bfloat16* dst ) {
  const uint16_t *p = reinterpret_cast<const uint16_t*>(src);
  uint16_t *q = reinterpret_cast<uint16_t*> (dst);
  *q = p[1];
}

inline __host__ __device__ void ToFloat32(const bfloat16* src, float* dst) {
  const uint16_t *p = reinterpret_cast<const uint16_t*>(src);
  uint16_t *q = reinterpret_cast<uint16_t*> (dst);
  q[0] = 0;
  q[1] = *p;
}

__global__ void convertFloat32ToBFloat16(const float* in, bfloat16* out, unsigned int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N) {
    ToBFloat16(in + index, out + index);
  }
}

__global__ void convertBFloat16ToFloat32(const bfloat16* in, float* out, unsigned int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N) {
    ToFloat32(in + index, out + index);
  }
}

float* allocInitF32(unsigned int memsize) {
  float *in = (float*) malloc (memsize);
  for (int i = 0; i < memsize/sizeof(float); i++) {
    in[i] = 0.12345678 + i;
  }
  return in;
}

bfloat16* allocInitBF16(unsigned int memsize) {
  bfloat16 *in = (bfloat16*) malloc (memsize);
  for (int i = 0; i < memsize/sizeof(bfloat16); i++) {
    float v = 0.12345678 + i;
    ToBFloat16(&v, &in[i]);
  }
  return in;
}


#define ARRAY_SIZE  8


int testConversionToBFloat16() {
  using IN_T = float;
  using OUT_T = bfloat16;
  
  unsigned int input_memsize = ARRAY_SIZE * sizeof(IN_T);
  unsigned int output_memsize = ARRAY_SIZE * sizeof(OUT_T);
  
  // initialize the input data
  IN_T *in = allocInitF32(input_memsize);
    
  // allocate the memory on the device side
  IN_T *in_GPU = nullptr;
  hipMalloc((void**)&in_GPU, input_memsize);

  OUT_T *out_GPU = nullptr;
  hipMalloc((void**)&out_GPU, output_memsize);

  // allocate memory on the host for the result
  OUT_T *out = (OUT_T*)malloc(output_memsize);

  // Memory transfer from host to device
  hipMemcpy(in_GPU, in, input_memsize, hipMemcpyHostToDevice);

  // Lauching kernel from host
  hipLaunchKernelGGL(convertFloat32ToBFloat16,
		  dim3(1), dim3(ARRAY_SIZE), 0, 0,
		  in_GPU, out_GPU, ARRAY_SIZE);

  // Memory transfer from device to host
  hipMemcpy(out, out_GPU, output_memsize, hipMemcpyDeviceToHost);

  // print the results
  for (int i = 0; i < ARRAY_SIZE; i++) {
    float in_f32 = in[i];
    bfloat16 out_bf16 = out[i];
    float out_f32 = 0;
    ToFloat32(&out_bf16, &out_f32);
    std::cout << in_f32 << " , " << out_f32 << std::endl;
  }

  // free the resources on device side
  hipFree(in_GPU);
  hipFree(out_GPU);

  // free the resources on host side
  free(in);
  free(out);

  return 0;
}

int testConversionFromBFloat16() {
  using IN_T = bfloat16;
  using OUT_T = float;
  
  unsigned int input_memsize = ARRAY_SIZE * sizeof(IN_T);
  unsigned int output_memsize = ARRAY_SIZE * sizeof(OUT_T);
  
  // initialize the input data
  IN_T *in = allocInitBF16(input_memsize);
    
  // allocate the memory on the device side
  IN_T *in_GPU = nullptr;
  hipMalloc((void**)&in_GPU, input_memsize);

  OUT_T *out_GPU = nullptr;
  hipMalloc((void**)&out_GPU, output_memsize);

  // allocate memory on the host for the result
  OUT_T *out = (OUT_T*)malloc(output_memsize);

  // Memory transfer from host to device
  hipMemcpy(in_GPU, in, input_memsize, hipMemcpyHostToDevice);

  // Lauching kernel from host
  hipLaunchKernelGGL(convertBFloat16ToFloat32,
		  dim3(1), dim3(ARRAY_SIZE), 0, 0,
		  in_GPU, out_GPU, ARRAY_SIZE);

  // Memory transfer from device to host
  hipMemcpy(out, out_GPU, output_memsize, hipMemcpyDeviceToHost);

  // print the results
  for (int i = 0; i < ARRAY_SIZE; i++) {
    bfloat16 in_bf16 = in[i];
    float in_f32 = 0; 
    ToFloat32(&in_bf16, &in_f32);
    float out_f32 = out[i];
    std::cout << in_f32 << " , " << out_f32 << std::endl;
  }

  // free the resources on device side
  hipFree(in_GPU);
  hipFree(out_GPU);

  // free the resources on host side
  free(in);
  free(out);

  return 0;
}

int main() {
  // testConversionToBFloat16();
  testConversionFromBFloat16();
  return 0;
}
