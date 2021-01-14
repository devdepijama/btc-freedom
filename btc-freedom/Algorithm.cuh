#include "Logger.cuh"
#include "Sha256.cuh"
#include "EllipticalCurve.cuh"

class Algorithm
{
private:
	Logger *logger;
	Sha256 hasher;
	EllipticalCurve ellipticalCurve;
	uint8_t* kernelBuffer;
	size_t kernelBufferSize;
	size_t kernelBlocks;
	size_t kernelThreads;

public:
	void init();
	Algorithm(const Sha256 &hasher, const EllipticalCurve &ellipticalCurve);
	void performAttack(unsigned int seed);

	~Algorithm();
};