#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
	int nDevices;

	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; ++i)
	{
		int memoryClockRate;
		cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, i);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("	Device name: %s\n", prop.name);
		printf("	Memory Clokc Rate (KHz): %d\n", memoryClockRate);
		printf("	Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("	Peak Memory Bindwidth (GB/s): %f\n", 2.0 * memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("	Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
		printf("	Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("	Number of SMs: %d\n", prop.multiProcessorCount);
		printf("	Max threads per block: %d\n", prop.maxThreadsPerBlock);

	}
}