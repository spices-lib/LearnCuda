#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory>
#include <cuda.h>

constexpr uint64_t SIZE = 1024 * 432 * 1024;

__global__ void test01()
{
	// print the blocks and threads IDs
	int warp_ID_Value = 0;
	warp_ID_Value = threadIdx.x / 32;
	printf("\nThe block ID is %d -- The thread ID is %d -- The warp id %d\n", blockIdx.x, threadIdx.x, warp_ID_Value);
}

__global__ void vectorAdd(int* A, int* B, int* C, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	C[i] = A[i] + B[i];
}

int main()
{
	int* A, * B, * C;
	int* d_A, * d_B, * d_C;
	int size = SIZE * sizeof(int);

	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	for (int i = 0; i < SIZE; ++i)
	{
		A[i] = i;
		B[i] = SIZE - i;
	}

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	vectorAdd <<<1024 * 432, 1024 >>> (d_A, d_B, d_C, SIZE);
	cudaEventRecord(stop);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("\nExecution finished, cost: %f milliseconds\n", milliseconds);
	/*for (int i = 0; i < SIZE; ++i)
	{
		printf("%d + %d = %d", A[i], B[i], C[i]);
		printf("\n");
	}*/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(A);
	free(B);
	free(C);

	cudaDeviceSynchronize();

	return 0;
}