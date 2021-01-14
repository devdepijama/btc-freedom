#include "Algorithm.cuh"

#include "device.cuh"
#include "constants.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Algorithm::Algorithm(const Sha256 &hasher, const EllipticalCurve &ellipticalCurve) : 
	hasher(hasher), 
	ellipticalCurve(ellipticalCurve)
{
	this->logger = new Logger("Algorithm", LOGGER_LEVEL_INFO);
	this->kernelBuffer = nullptr;
	this->kernelBufferSize = KERNEL_RESULT_BUFFER_SIZE;
	this->kernelBlocks = KERNEL_BLOCKS;
	this->kernelThreads = KERNEL_THREADS;
}

void Algorithm::init() {
	this->logger->info("Opening default CUDA device...");
	cudaError_t rv = cudaSetDevice(CUDA_DEFAULT_DEVICE);
	if (rv != ::cudaSuccess) {
		this->logger->error("Could not open default device... Status: %d", rv);
		return;
	}

	rv = cudaMalloc(&(this->kernelBuffer), this->kernelBufferSize);
	this->logger->info("Allocating %lu bytes for GPU processing...", this->kernelBufferSize);
	if (rv != ::cudaSuccess) {
		this->logger->error("Could not allocate %lu bytes for GPU processing. Status: %d", this->kernelBufferSize, rv);
		return;
	}
}

void Algorithm::performAttack(unsigned int seed) {
	this->logger->info("Performing attack...");
	
	const size_t totalThreads = this->kernelBlocks * this->kernelThreads;
	this->logger->info("Invoking GPU Kernel with %d blocks and %d threads each. Total = %d", this->kernelBlocks, this->kernelThreads, totalThreads);

	kernel <<< 1, 1>>> (this->kernelBuffer, this->kernelBufferSize);

	cudaError_t rv = cudaDeviceSynchronize();
	if (rv != cudaSuccess) {
		this->logger->error("Could not synchronize with device. Error code %d", rv);
		return;
	}
}

Algorithm::~Algorithm() {
	this->logger->info("Closing default device...");
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != ::cudaSuccess) {
		this->logger->error("Could not close default device... Status: %d", cudaStatus);
		return;
	}

	this->logger->info("Freeing reserved bytes on GPU");
	cudaFree(this->kernelBuffer);

	delete this->logger;
}