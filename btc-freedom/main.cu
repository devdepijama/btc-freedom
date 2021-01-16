#include "Algorithm.cuh"

#include <stdio.h>

int main() {
    Algorithm algorithm = Algorithm();
    
    algorithm.init();
    algorithm.performAttack(1);

    return 0;
}