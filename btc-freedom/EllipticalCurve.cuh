#ifndef H_ELLIPTICAL_CURVE
#define H_ELLIPTICAL_CURVE

#include <stdint.h>

class EllipticalCurve
{
	public:
		static int calculatePublicKey(uint8_t* data, size_t len, uint8_t* output);
};

#endif