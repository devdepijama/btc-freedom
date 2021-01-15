#ifndef H_ELLIPTICAL_CURVE
#define H_ELLIPTICAL_CURVE

#include <stdint.h>
#include <secp256k1.h>

class EllipticalCurve
{
	private:
		secp256k1_context* ctx;
	public:
		EllipticalCurve();
		int calculatePublicKey(uint8_t* data, size_t len, uint8_t* output);
		~EllipticalCurve();
};

#endif