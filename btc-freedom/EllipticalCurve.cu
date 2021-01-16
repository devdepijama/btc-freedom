#include "EllipticalCurve.cuh"

#include <string.h>
#include <stdlib.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#define SECP256K1_PUB_KEY_STRING_SIZE 67

int EllipticalCurve::calculatePublicKey(char* hexPrivateKey, char* hexPublicKey) {
    // Set things ups
    EC_KEY* ecKey = EC_KEY_new_by_curve_name(NID_secp256k1);
    const EC_GROUP* ecGroupSecp256k1 = EC_KEY_get0_group(ecKey);

    // Set private key
    BIGNUM* bnPrivateKey = BN_new();
    BN_hex2bn(&bnPrivateKey, hexPrivateKey);
    EC_KEY_set_private_key(ecKey, bnPrivateKey);

    // Get public key
    BN_CTX* bnCtx = BN_CTX_new();
    EC_POINT* publicKeyPoint = EC_POINT_new(ecGroupSecp256k1);
    EC_POINT_mul(ecGroupSecp256k1, publicKeyPoint, bnPrivateKey, NULL, NULL, bnCtx);
    EC_KEY_set_public_key(ecKey, publicKeyPoint);
    
    char* result = EC_POINT_point2hex(
        ecGroupSecp256k1, 
        publicKeyPoint, 
        POINT_CONVERSION_COMPRESSED,
        bnCtx
    );
    for (int i = 0; i < SECP256K1_PUB_KEY_STRING_SIZE; i++) {
        hexPublicKey[i] = (result[i] >= 'A') ? (result[i] - 'A') + 'a' : result[i];
    }

    EC_KEY_free(ecKey);
    BN_CTX_free(bnCtx);
    BN_free(bnPrivateKey);
    free(result);

	return 0;
}