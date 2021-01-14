#include "Logger.cuh"
#include "Sha256.cuh"
#include "EllipticalCurve.cuh"
#include "Algorithm.cuh"

#include <stdio.h>

int main()
{
    Sha256 sha256 = Sha256();
    EllipticalCurve ellipticalCurve = EllipticalCurve();
    Algorithm algorithm = Algorithm(sha256, ellipticalCurve);
    
    algorithm.init();
    algorithm.performAttack(1);

    return 0;
}