#ifndef H_RIPEMD160
#define H_RIPEMD160

#include <stdint.h>
#include <openssl/ripemd.h>

class Ripemd160 {
	public:
		static int hash(uint8_t* data, size_t len, uint8_t output[RIPEMD160_DIGEST_LENGTH]);
};

#endif
