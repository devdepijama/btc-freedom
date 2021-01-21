#include "Logger.cuh"
#include "Sha256.cuh"
#include "Ripemd160.cuh"
#include "EllipticalCurve.cuh"
#include "Base58.cuh"

class Algorithm
{
private:
	Logger *logger;
	uint8_t* kernelBuffer;
	size_t kernelBufferSize;
	size_t kernelBlocks;
	size_t kernelThreads;

public:
	void init();
	Algorithm();
	void performAttack(int seed);

	~Algorithm();
};