#ifndef H_ELLIPTICAL_CURVE
#define H_ELLIPTICAL_CURVE

#include <stdint.h>

class EllipticalCurve
{
	public:
		static int calculatePublicKey(char *hexPrivateKey, char* hexPublicKey);
};

#endif