/*
=================================================
			Adithya H K Upadhya
			MS CS, Virginia Tech
			adithyau@vt.edu
			4/25/2018
=================================================
*/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)


enum WeightingFunction { GAUSSIAN_WEIGHTING = 0, L1_NORM = 1, L2_NORM = 2, L3_NORM = 3, L4_NORM = 4 };

#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess) {                                                              \
            printf("\nCuda failure file - %s\nLine - %d\nErrorStr - %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                                    \
    }


void launchParallelNLM(float *noisyData, float *kernel, float *denoisedImage,
	int f, int t, float h, int sizeX, int sizeY, int sizeZ,
	WeightingFunction weight);