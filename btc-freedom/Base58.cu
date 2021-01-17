#include "Base58.cuh"

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <openssl/bn.h>

using namespace std;

string Base58::cypher(char* hexBytes) {
    char table[] = { '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z' };

    //printf("Calculating b58 of %s \n", hexBytes);

    BIGNUM* base58 = NULL;

    BIGNUM* resultExp = BN_new();
    BIGNUM* resultAdd = BN_new();
    BIGNUM* resultRem = BN_new();
    BN_CTX* bn_ctx = BN_CTX_new();

    BN_dec2bn(&base58, "58");

    string endresult;
    vector<int> v;

    BN_hex2bn(&resultAdd, hexBytes);

    while (!BN_is_zero(resultAdd)) {
        BN_div(resultAdd, resultRem, resultAdd, base58, bn_ctx);
        //printf("resultAdd = %s | resultRem = %s \n", BN_bn2dec(resultRem), BN_bn2dec(resultAdd));
        v.push_back(atoi(BN_bn2dec(resultRem)));
    }

    for (int i = 0; i < strlen(hexBytes);) {
        if ((hexBytes[i] == '0') && (hexBytes[i + 1] == '0')) {
            endresult = endresult + '1'; 
            i += 2;
        }
        break;        
    }

    for (int i = (int)v.size() - 1; i >= 0; i--) {
        endresult = endresult + table[v[i]];
    }

    BN_free(resultAdd);
    BN_free(resultExp);
    BN_free(resultRem);
    BN_CTX_free(bn_ctx);

    return endresult;
}
