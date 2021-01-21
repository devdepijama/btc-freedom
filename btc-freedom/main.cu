#include "Algorithm.cuh"

#include <stdio.h>

int main() {
    Algorithm algorithm = Algorithm();
    int seed = 0;
    
    algorithm.init();

    do {
        algorithm.performAttack(seed++);
    } while (seed < 1000);

    return 0;
}