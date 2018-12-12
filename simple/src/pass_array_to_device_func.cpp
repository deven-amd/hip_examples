#include <iostream>
#include <hip/hip_runtime.h>

template <typename Scalar, int N>
struct chebevl {
  static __host__ __device__ __inline__ Scalar run(Scalar x, const Scalar coef[]) {
    Scalar b0 = coef[0];
    Scalar b1 = 0;
    Scalar b2;

    for (int i = 1; i < N; i++) {
      b2 = b1;
      b1 = b0;
      b0 = x * b1 - b2 + coef[i];
    }

    return Scalar(0.5) * (b0 - b2);
  }
};

float cpu_kernel(float input) {

  const float B[] = {3.39623202570838634515E-9f, 2.26666899049817806459E-8f,
		     2.04891858946906374183E-7f, 2.89137052083475648297E-6f,
		     6.88975834691682398426E-5f, 3.36911647825569408990E-3f,
		     8.04490411014108831608E-1f};
  
  return chebevl<float, 7>::run(32.0f / input - 2.0f, B);
}

__global__ void gpu_kernel(hipLaunchParm lp, float* output, float* input) {

  const float B[] = {3.39623202570838634515E-9f, 2.26666899049817806459E-8f,
  		     2.04891858946906374183E-7f, 2.89137052083475648297E-6f,
  		     6.88975834691682398426E-5f, 3.36911647825569408990E-3f,
  		     8.04490411014108831608E-1f};

  int tid = threadIdx.x;
  float x = input[tid];
  output[tid] = chebevl<float, 7>::run(32.0f / x - 2.0f, B);
  //output[tid] = 42;
}

float gpu_kernel_caller(float input) {

  const int N = 16;
  auto memsize = N*sizeof(float);

  float* inputCPU = new float[N];
  float* outputCPU = new float[N];

  for (int i=0; i<N; i++) {
    inputCPU[i] = input;
    outputCPU[i] = -1;
  }
  float* inputGPU = nullptr;
  float* outputGPU = nullptr;
  
  hipError_t status = hipMalloc((void**)&inputGPU, memsize);
  if (status != hipSuccess) {
    std::cout << "call to hipMalloc (inputGPU) failed!" << std::endl;
  }

  status = hipMalloc((void**)&outputGPU, memsize);
  if (status != hipSuccess) {
    std::cout << "call to hipMalloc (outputGPU) failed!" << std::endl;
  }

  // copy input from host
  status = hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice);
  if (status != hipSuccess) {
    std::cout << "call to hipMemcpy (input) failed!" << std::endl;
  }

  // launch the kernel
  hipLaunchKernel(gpu_kernel, dim3(1), dim3(N), 0, 0, outputGPU, inputGPU);

  hipDeviceSynchronize();
  
  // copy output from device
  status = hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    std::cout << "call to hipMemcpy (output) failed!" << std::endl;
  }

  float answer = outputCPU[0];
  
  // free memory
  delete [] inputCPU;
  delete [] outputCPU;
  hipFree(inputGPU);
  hipFree(outputGPU);

  return answer;
}

int main() {

  float x = 20;
  
  float cpu_answer = cpu_kernel(x);
  float gpu_answer = gpu_kernel_caller(x);
  
  std::cout << "cpu_answer = " <<  cpu_answer << std::endl;
  std::cout << "gpu_answer = " <<  gpu_answer << std::endl;

  return 0;
}
