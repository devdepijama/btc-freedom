#ifndef H_SHA256
#define H_SHA256

#include <stdint.h>

class Sha256
{
	public:
		static int hash(uint8_t* data, size_t len, uint8_t* output);
};

#endif