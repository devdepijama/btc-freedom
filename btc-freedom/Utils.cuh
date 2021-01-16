#ifndef H_UTILS
#define H_UTILS

#include <stdint.h>

class Utils {
	public:
		static void bytes_to_hex(uint8_t* bytes, size_t bytes_len, char* hex, size_t hex_len);
		static void hex_to_bytes(char* hex, uint8_t* bytes, size_t bytes_len);
};

#endif