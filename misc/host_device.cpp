// /opt/rocm/bin/hipcc -std=c++11 host_device.cpp

#include <hip/hip_runtime.h>

                    float plain_func(float a) { return a+2; }
__host__            float host_func(float a) { return a+3; }
__device__          float device_func(float a) { return a+4; }
__host__ __device__ float host_device_func(float a) { return a+5; }


__device__ static float test_device_from_gpu(float a) {

  float v1, v2, v3, v4;
  
  v1 = plain_func(a);
  // v2 = host_func(a);   // call from AMP-restricted function to CPU-restricted function
  v3 = device_func(a);
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

__host__ __device__ static float test_host_device_from_gpu(float a) {

  float v1, v2, v3, v4;
  
  v1 = plain_func(a);
  // v2 = host_func(a);   // call from AMP-restricted function to CPU-restricted function
  v3 = device_func(a);
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

__host__ __device__ static float test_host_device_from_both(float a) {

  float v1, v2, v3, v4;
  
  v1 = plain_func(a);
  v2 = host_func(a);   // call from AMP-restricted function to CPU-restricted function
  v3 = device_func(a);
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

__global__ void global_func(hipLaunchParm lp, float a) {

  // plain_func(a); // 'plain_func': no overloaded function has restriction specifiers that are compatible with the ambient context 'global_func'
  // host_func(a);   // call from AMP-restricted function to CPU-restricted function
  device_func(a);
  host_device_func(a);

  test_device_from_gpu(a);
  
  test_host_device_from_gpu(a);

  test_host_device_from_both(a);
}

static float test_plain_from_cpu(float a) {

  float v1, v2, v3, v4;

  v1 = plain_func(a);
  v2 = host_func(a);
  // v3 = device_func(a);  // 'device_func': no overloaded function has restriction specifiers that are compatible with the ambient context 'test_cpu'
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

__host__ static float test_host_from_cpu(float a) {

  float v1, v2, v3, v4;
  
  v1 = plain_func(a);
  v2 = host_func(a);
  // v3 = device_func(a);  // 'device_func': no overloaded function has restriction specifiers that are compatible with the ambient context 'test_cpu'
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

__host__ __device__ static float test_host_device_from_cpu(float a) {

  float v1, v2, v3, v4;
  
  v1 = plain_func(a);
  // v2 = host_func(a); // call from AMP-restricted function to CPU-restricted function
  v3 = device_func(a);  // /tmp/host_device-84b0bc.o: In function `test_cpu()':  host_device.cpp:(.text+0x39a4): undefined reference to `device_func()'
  v4 = host_device_func(a);

  return v1+v2+v3+v4;
}

static float test_cpu(float a) {

  float v1, v2, v3, v4;
  
  v1 = test_plain_from_cpu(a);
  v2 = test_host_from_cpu(a);
  v3 = test_host_device_from_cpu(a);
  v4 = test_host_device_from_both(a);

  return v1+v2+v3+v4;
}

int main(int argc, const char** argv) {
  float a = atof(argv[1]);
  test_cpu(a);
  hipLaunchKernel(global_func, dim3(1), dim3(1), 0, 0, a);
  return 0;
}


