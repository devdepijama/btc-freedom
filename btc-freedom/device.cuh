#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>

__global__ void kernel(uint8_t* buffer, size_t bufferLen);