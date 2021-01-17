#include "Ripemd160.cuh"

#include <openssl/ripemd.h>


int Ripemd160::hash(uint8_t* data, size_t len, uint8_t output[RIPEMD160_DIGEST_LENGTH]) {

	RIPEMD160_CTX ctx;

	RIPEMD160_Init(&ctx);
	RIPEMD160_Update(&ctx, data, len);
	RIPEMD160_Final(output, &ctx);

	return 0;
}