#include "Sha256.cuh"

Sha256::Sha256() {

}

int Sha256::hash(uint8_t* data, size_t len, uint8_t output[SHA256_DIGEST_LENGTH]) {
    SHA256_Init(&(this->sha256));
    SHA256_Update(&(this->sha256), data, len);
    SHA256_Final(output, &sha256);

    return 0;
}

Sha256::~Sha256() {

}
