#include "Utils.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static char nibble_to_char(uint8_t nibble) {
	return (nibble < 10) ? nibble + '0' : (nibble - 10) + 'a';
}

static uint8_t char_to_nibble(char character) {
	if (('a' <= character) && (character <= 'f')) return character - 'a';
	return character - '0';
}

void Utils::bytes_to_hex(uint8_t* bytes, size_t bytes_len, char* hex, size_t hex_len) {
	if (((bytes_len << 1) + 1) > hex_len) return;

	int j = 0;
	for (int i = 0; i < bytes_len; i++) {
		hex[j++] = nibble_to_char(bytes[i] >> 4);
		hex[j++] = nibble_to_char(bytes[i] & 0x0F);
	}

	hex[j++] = '\0';
}

void Utils::hex_to_bytes(char* hex, uint8_t* bytes, size_t bytes_len) {
	size_t hex_len = strlen(hex);
	if (bytes_len < (hex_len << 1)) return;

	int j = 0;
	for (int i = 0; i < hex_len; i++) {
		bytes[j++] = char_to_nibble(hex[i] >> 4);
		bytes[j++] = char_to_nibble(hex[i] & 0x0F);
	}
}