#include "Algorithm.cuh"

#include "device.cuh"
#include "constants.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "Utils.cuh"

Algorithm::Algorithm() {
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

	// Print bytes that will work as seed:
	char hexSeed[9];
	uint8_t to_be_hashed[] = { 0x00, 0x00, 0x00, 0x01 };
	Utils::bytes_to_hex(to_be_hashed, sizeof(to_be_hashed), hexSeed, sizeof(hexSeed));
	this->logger->info("Seed: %s", hexSeed);

	// Calculate SHA256 of it:
	uint8_t hash[SHA256_DIGEST_LENGTH];
	Sha256::hash(to_be_hashed, sizeof(to_be_hashed), hash);

	char hexPrivateKey[65];
	Utils::bytes_to_hex(hash, sizeof(hash), hexPrivateKey, sizeof(hexPrivateKey));
	this->logger->info("PrivateKey - SHA256(Seed): %s", hexPrivateKey);

	// Calculate the secp256k1
	char hexPublicKey[67];
	EllipticalCurve::calculatePublicKey(hexPrivateKey, hexPublicKey);
	this->logger->info("PublicKey - secp256k1(PrivateKey): %s", hexPublicKey);

	// Convert hex to bytes
	uint8_t bytesPublicKey[132];
	Utils::hex_to_bytes(hexPublicKey, bytesPublicKey, sizeof(bytesPublicKey));
	Sha256::hash(bytesPublicKey, sizeof(bytesPublicKey), hash);

	char hexSHA256PubKey[65];
	Utils::bytes_to_hex(hash, sizeof(hash), hexSHA256PubKey, sizeof(hexSHA256PubKey));
	this->logger->info("SHA256(PublicKey): %s", hexSHA256PubKey);

	//kernel <<< 1, 1>>> (this->kernelBuffer, this->kernelBufferSize);

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