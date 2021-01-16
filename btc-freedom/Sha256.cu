#include "Sha256.cuh"

#include <openssl/sha.h>

int Sha256::hash(uint8_t* data, size_t len, uint8_t output[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX sha256;

    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, len);
    SHA256_Final(output, &sha256);

    return 0;
}
