#include "device.cuh"

__global__ void kernel(uint8_t* buffer, size_t bufferLen) {
	int threadId = threadIdx.x;

	for (size_t i = 0; i < bufferLen; i++) {
		buffer[i] = threadId;
	}
}