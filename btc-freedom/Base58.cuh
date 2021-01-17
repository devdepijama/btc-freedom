#ifndef H_BASE58
#define H_BASE58

#include <stdint.h>
#include <string>

using namespace std;

class Base58 {
	public:
		static string Base58::cypher(char* hexBytes);
};

#endif