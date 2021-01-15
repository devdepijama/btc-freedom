#include "EllipticalCurve.cuh"

#include <stdlib.h>

EllipticalCurve::EllipticalCurve() {
	this->ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
}

int EllipticalCurve::calculatePublicKey(uint8_t* data, size_t len, uint8_t* output) {
	secp256k1_ec_pubkey_create(this->ctx, (secp256k1_pubkey *) output, data);
	return 0;
}

EllipticalCurve::~EllipticalCurve() {
}
