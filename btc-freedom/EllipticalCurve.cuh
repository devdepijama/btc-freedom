#ifndef H_ELLIPTICAL_CURVE
#define H_ELLIPTICAL_CURVE

#include <stdint.h>
#include "Logger.cuh"

class EllipticalCurve
{
	private:
		Logger *logger;

	public:
		EllipticalCurve();
		int calculatePublicKey(char *hexPrivateKey, char* hexPublicKey);
		~EllipticalCurve();
};

#endif