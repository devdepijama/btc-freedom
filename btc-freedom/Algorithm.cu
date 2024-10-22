#include "Algorithm.cuh"
#include "Utils.cuh"

#include "device.cuh"
#include "constants.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void Algorithm::performAttack(int seed) {
	this->logger->info("#############################################################");
	this->logger->info("Performing attack...");
	
	const size_t totalThreads = this->kernelBlocks * this->kernelThreads;
	this->logger->info("Invoking GPU Kernel with %d blocks and %d threads each. Total = %d", this->kernelBlocks, this->kernelThreads, totalThreads);

	// Print bytes that will work as seed:
	char hexSeed[9];
	uint8_t bytesSeed[4];
	memcpy(bytesSeed, &seed, sizeof(seed));
	Utils::bytes_to_hex(bytesSeed, sizeof(bytesSeed), hexSeed, sizeof(hexSeed));
	this->logger->info("Seed: %s", hexSeed);

	// Calculate SHA256 of it:
	char hexPrivateKey[65];
	uint8_t bytesPrivateKey[SHA256_DIGEST_LENGTH];

	Sha256::hash(bytesSeed, sizeof(bytesSeed), bytesPrivateKey);
	Utils::bytes_to_hex(bytesPrivateKey, sizeof(bytesPrivateKey), hexPrivateKey, sizeof(hexPrivateKey));
	this->logger->info("PrivateKey - SHA256(Seed): %s", hexPrivateKey);

	// Calculate the secp256k1
	char hexPublicKey[67];
	uint8_t bytesPublicKey[33];

	EllipticalCurve::calculatePublicKey(hexPrivateKey, hexPublicKey);
	Utils::hex_to_bytes(hexPublicKey, bytesPublicKey, sizeof(bytesPublicKey));
	this->logger->info("PublicKey - secp256k1(PrivateKey): %s", hexPublicKey);

	// Calculate SHA256 of public key
	char hexSHA256PublicKey[65];
	uint8_t bytesSHA256PublicKey[SHA256_DIGEST_LENGTH];

	Sha256::hash(bytesPublicKey, sizeof(bytesPublicKey), bytesSHA256PublicKey);
	Utils::bytes_to_hex(bytesSHA256PublicKey, sizeof(bytesSHA256PublicKey), hexSHA256PublicKey, sizeof(hexSHA256PublicKey));
	this->logger->info("SHA256(PublicKey): %s", hexSHA256PublicKey);

	// Apply RIPMD160 on sha256(publicKey)
	char hexRIPMD160[41];
	uint8_t bytesRIPMD160[RIPEMD160_DIGEST_LENGTH];

	Ripemd160::hash(bytesSHA256PublicKey, sizeof(bytesSHA256PublicKey), bytesRIPMD160);
	Utils::bytes_to_hex(bytesRIPMD160, sizeof(bytesRIPMD160), hexRIPMD160, sizeof(hexRIPMD160));
	this->logger->info("RIPMD160(SHA256(PublicKey)): %s", hexRIPMD160);

	// Append 0x00 (Main Network) 
	char hexWithMainNetworkAppended[43];
	uint8_t bytesWithMainNetworkAppended[RIPEMD160_DIGEST_LENGTH + 1];

	bytesWithMainNetworkAppended[0] = 0x00;
	memcpy(bytesWithMainNetworkAppended + sizeof(uint8_t), bytesRIPMD160, sizeof(bytesRIPMD160));
	Utils::bytes_to_hex(bytesWithMainNetworkAppended, sizeof(bytesWithMainNetworkAppended), hexWithMainNetworkAppended, sizeof(hexWithMainNetworkAppended));
	this->logger->info("0x00 + RIPMD160(SHA256(PublicKey)): %s", hexWithMainNetworkAppended);

	// Calculate SHA256 two times on the previous result
	char hexDoubleSHA256[65];
	uint8_t bytesDoubleSHA256[SHA256_DIGEST_LENGTH];

	Sha256::hash(bytesWithMainNetworkAppended, sizeof(bytesWithMainNetworkAppended), bytesDoubleSHA256);
	Utils::bytes_to_hex(bytesDoubleSHA256, sizeof(bytesDoubleSHA256), hexDoubleSHA256, sizeof(hexDoubleSHA256));
	this->logger->info("SHA256(0x00 + RIPMD160(SHA256(PublicKey))): %s", hexDoubleSHA256);

	Sha256::hash(bytesDoubleSHA256, sizeof(bytesDoubleSHA256), bytesDoubleSHA256);
	Utils::bytes_to_hex(bytesDoubleSHA256, sizeof(bytesDoubleSHA256), hexDoubleSHA256, sizeof(hexDoubleSHA256));
	this->logger->info("SHA256(SHA256(0x00 + RIPMD160(SHA256(PublicKey)))): %s", hexDoubleSHA256);

	// Append previous hash to the end
	uint8_t bytesResult[sizeof(bytesWithMainNetworkAppended) + 4];
	char hexResult[(2 * sizeof(bytesResult)) + 1];

	// Glue previous parts together
	memcpy(bytesResult, bytesWithMainNetworkAppended, sizeof(bytesWithMainNetworkAppended));
	memcpy(bytesResult + sizeof(bytesWithMainNetworkAppended), bytesDoubleSHA256, 4);
	Utils::bytes_to_hex(bytesResult, sizeof(bytesResult), hexResult, sizeof(hexResult));
	this->logger->info("Result before base58: %s", hexResult);

	// Result
	this->logger->info("Private Key (Base58): %s ", Base58::cypher(hexPrivateKey).c_str());
	this->logger->info("Address (Base58): %s ", Base58::cypher(hexResult).c_str());
	this->logger->info("#############################################################");

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