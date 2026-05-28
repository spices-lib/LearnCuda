
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

__global__ void reduce_in_place(float* input, int n) {

	__shared__ float shared[256];

	int tid = threadIdx.x;
	int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	shared[tid] = input[index] + input[index + blockDim.x];

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
	
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}

		__syncthreads();
	}

	if (tid == 0) {
		input[blockIdx.x] = input[blockIdx.x * blockDim.x];
	}
}

__global__ void reduce_with_warp_optimization(float* input, int n) {

	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;

	sum = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);

	for (int offset = warpSize / 2; offset > 0; offset >> 1) {
		sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
	}

	if (tid % warpSize == 0) {
		shared[tid / warpSize] = sum;
	}

	__syncthreads();

	if (tid < warpSize) {
		sum = (tid < (blockDim.x / warpSize)) ? shared[tid] : 0.0f;
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
		}
	}

	if (tid == 0) {
		input[blockIdx.x] = sum;
	}
}

float cpu_reduce(float* input, int n) {

	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += input[i];
	}
	return sum;
}

int main() {

	int n = 1024 * 1024;
	size_t size = n * sizeof(float);
	float* d_input;
	float* h_input = new float[n];
	cudaMalloc(&d_input, size);

	for (int i = 0; i < n; i++) {
		h_input[i] = static_cast<float>(i);
	}

	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;
	std::cout << gridSize << std::endl;

	float sum = cpu_reduce(h_input, n);
	std::cout << "CPU sum   " << sum << std::endl;

	while (gridSize > 1)
	{
		reduce_with_warp_optimization << <gridSize, blockSize >> > (d_input, n);
		cudaDeviceSynchronize();

		n = gridSize;
		gridSize = (n + blockSize - 1) / blockSize;
	}

	reduce_with_warp_optimization <<<1, blockSize>>> (d_input, n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "GPU output    " << h_input[0] << std::endl;

	cudaFree(d_input);
	delete[] h_input;
	return 0;
}