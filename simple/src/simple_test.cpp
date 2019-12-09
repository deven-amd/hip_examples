
#include <iostream>

// hip header file
#include "hip/hip_runtime.h"

struct GpuDeviceInfo {

  bool init_success;
  
  int num_gpus;
  hipDeviceProp_t *gpu_properties;
  
  GpuDeviceInfo() {
    init_success = false;
    hipError_t status = hipGetDeviceCount(&num_gpus);
    if (status != hipSuccess) {
      std::cout << "\tCall to hipGetDeviceCount failed!\n";
    } else {
      gpu_properties = new hipDeviceProp_t[num_gpus];
      for (int i=0; i<num_gpus; i++) {
	status = hipGetDeviceProperties(&gpu_properties[i], i);
	if (status != hipSuccess) {
	  std::cout << "\tCall to hipGetDeviceProperties failed for gpu #" << i << "!\n";
	}
      }
    }
    init_success = true;
  }

  ~GpuDeviceInfo() {
    if (init_success) {
      delete [] gpu_properties;
    }
  }
};

template<int NumDims>
struct GpuKernelLaunchInfo;

template<>
struct GpuKernelLaunchInfo<1> {

  dim3 threads_per_block;
  dim3 blocks_per_grid;
  dim3 step_size;
  
  GpuKernelLaunchInfo(int N, hipDeviceProp_t device_info) {
    threads_per_block = dim3(device_info.maxThreadsPerBlock, 1, 1);

    uint32_t logical_blocks = (N + threads_per_block.x - 1) / threads_per_block.x;
    uint32_t physical_blocks = (device_info.multiProcessorCount * device_info.maxThreadsPerMultiProcessor) /  threads_per_block.x;
    blocks_per_grid = dim3(std::min(logical_blocks, physical_blocks), 1, 1);

    step_size = dim3(blocks_per_grid.x * threads_per_block.x, 1, 1);
    
    // std::cout << "Kernel Launch Config: \n";
    // std::cout << "\tN : " << N << "\n";
    // std::cout << "\tthreads_per_block : " << threads_per_block.x << "\n";
    // std::cout << "\tblocks_per_grid : " << blocks_per_grid.x << "\n";
    // std::cout << "\tstep_size: " << step_size.x << "\n";
  }

  ~GpuKernelLaunchInfo() {}
};

template<typename T>
__global__ void gpu_kernel_1D(uint32_t N, uint32_t S, T* out0, T* in0, T* in1) {

  uint32_t first_index = blockDim.x * blockIdx.x + threadIdx.x;
  // uint32_t step_size = gridDim.x * blockDim.x;
  uint32_t step_size = S;

  for (uint32_t index = first_index; index < N; index += step_size) {
    out0[index] = in0[index] * in1[index];
  }
}

template<typename T, int N>
struct TestcaseData {

  size_t memsize;

  T *in0_host;
  T *in1_host;
  T *out0_host;
  
  T *in0_device;
  T *in1_device;
  T *out0_device;
  
  TestcaseData() {
    memsize = N * sizeof(T);
    
    in0_host = (T*) malloc (memsize);
    in1_host = (T*) malloc (memsize);
    out0_host = (T*) malloc (memsize);
    
    hipMalloc((void**)&in0_device, memsize);
    hipMalloc((void**)&in1_device, memsize);
    hipMalloc((void**)&out0_device, memsize);
  }
  
  ~TestcaseData() {
    free(in0_host);
    free(in1_host);
    free(out0_host);
    
    hipFree(in0_device);
    hipFree(in1_device);
    hipFree(out0_device);
  }

  void initialize_inputs() {
    for (int i = 0; i < N; i++) {
      in0_host[i] = 2.0*i;
      in1_host[i] = 0.5;
    }
  }

  void copy_inputs_to_device() {
    hipMemcpy(in0_device, in0_host, memsize, hipMemcpyHostToDevice);
    hipMemcpy(in1_device, in1_host, memsize, hipMemcpyHostToDevice);
  }

  void copy_outputs_from_device() {
    hipMemcpy(out0_host, out0_device, memsize, hipMemcpyDeviceToHost);
  }

  int check_outputs() {
    for (int i = 0; i < N; i++) {
      T out0 = out0_host[i];
      if (int(out0) != i) {
	
	std::cout << "\n\nFAIL (" << int(out0) << ", " << i << ")\n\n\n";
	return 1;
      }
    }
    std::cout << "\n\nPASS\n\n\n";
    return 0;
  }
};

template<typename T, int N, int M>
int simpleTest() {

  GpuDeviceInfo gpus;

  GpuKernelLaunchInfo<1> launch_info(N, gpus.gpu_properties[0]);
  
  TestcaseData<float,N> test_data;

  test_data.initialize_inputs();

  test_data.copy_inputs_to_device();

  // Lauching kernel from host
  for (int i=0; i <M; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_kernel_1D<T>),
		       launch_info.blocks_per_grid, launch_info.threads_per_block, 0, 0,
		       N, launch_info.step_size.x, test_data.out0_device, test_data.in0_device, test_data.in1_device);
  }

  test_data.copy_outputs_from_device();

  return test_data.check_outputs();
}


int main() {
  return simpleTest<float, 1024*512, 10>();
}
