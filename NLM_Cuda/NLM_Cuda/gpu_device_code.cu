/*
=================================================
			Adithya H K Upadhya
			MS CS, Virginia Tech
			adithyau@vt.edu
			4/25/2018
=================================================
*/

#include "gpu_device_code.cuh"

__global__ void parallelGaussianNLM(float *__restrict noisy, float *__restrict kernel, float *__restrict denoisedImage,
	const int f, const int t, const float h,
	const int sizeX, const int sizeY, const int sizeZ) {

	const int x = f + threadIdx.x + blockIdx.x * blockDim.x,
		y = f + threadIdx.y + blockIdx.y * blockDim.y,
		z = f + threadIdx.z + blockIdx.z * blockDim.z,
		denoiseIndex = (z - f) * sizeY * sizeX + (x - f) * sizeY + (y - f);

	if (x >= (sizeX + f) || y >= (sizeY + f) || z >= (sizeZ + f)) return;

	denoisedImage[denoiseIndex] = 0;

	float normalize = 0, maximum = 0;

	// Iterating over the Search Window of the voxel.
	// Boundary checks applied here through MAX & MIN macros.
	for (int zSearch = MAX(z - t, f); zSearch <= MIN(z + t, sizeZ + f - 1); zSearch++)
		for (int ySearch = MAX(y - t, f); ySearch <= MIN(y + t, sizeY + f - 1); ySearch++)
			for (int xSearch = MAX(x - t, f); xSearch <= MIN(x + t, sizeX + f - 1); xSearch++) {

				if (xSearch == x && ySearch == y && zSearch == z) continue;

				float d = 0;
				// Iterating over the local window of a voxel
				for (int lz = -f; lz <= f; lz++)
					for (int ly = -f; ly <= f; ly++)
						for (int lx = -f; lx <= f; lx++) {

							const int searchIndex = (zSearch + lz) * (sizeY + 2 * f) * (sizeX + 2 * f) +
								(xSearch + lx) * (sizeY + 2 * f) + (ySearch + ly),

								targetIndex = (z + lz) * (sizeY + 2 * f) * (sizeX + 2 * f) +
								(x + lx) * (sizeY + 2 * f) + (y + ly),

								kernelIndex = (lz + f) * (2 * f + 1) * (2 * f + 1) +
								(lx + f) * (2 * f + 1) + (ly + f);


							d += kernel[kernelIndex] * (noisy[searchIndex] - noisy[targetIndex]) *
								(noisy[searchIndex] - noisy[targetIndex]);


						}

				d = expf(-d / (h*h));
				normalize += d;
				maximum = MAX(maximum, d);

				denoisedImage[denoiseIndex] += d * noisy[zSearch * (sizeY + 2 * f) * (sizeX + 2 * f) +
					xSearch * (sizeY + 2 * f) + ySearch];
			}

	denoisedImage[denoiseIndex] += maximum * noisy[z * (sizeY + 2 * f) * (sizeX + 2 * f) + x * (sizeY + 2 * f) + y];

	denoisedImage[denoiseIndex] /= (normalize + maximum);

}



__global__ void parallelLkNormNLM(float *__restrict noisy, float *__restrict denoisedImage,
	const int f, const int t,
	const int sizeX, const int sizeY, const int sizeZ, const int weight) {

	const int x = f + threadIdx.x + blockIdx.x * blockDim.x,
		y = f + threadIdx.y + blockIdx.y * blockDim.y,
		z = f + threadIdx.z + blockIdx.z * blockDim.z,
		denoiseIndex = (z - f) * sizeY * sizeX + (x - f) * sizeY + (y - f);

	if (x >= (sizeX + f) || y >= (sizeY + f) || z >= (sizeZ + f)) return;

	denoisedImage[denoiseIndex] = 0;

	float X = 0, Y = 0, minimum = 3.4e38;

	int count = -1;

	// Iterating over the Search Window of the voxel.
	// Boundary checks applied here through MAX & MIN macros.
	for (int zSearch = MAX(z - t, f); zSearch <= MIN(z + t, sizeZ + f - 1); zSearch++)
		for (int ySearch = MAX(y - t, f); ySearch <= MIN(y + t, sizeY + f - 1); ySearch++)
			for (int xSearch = MAX(x - t, f); xSearch <= MIN(x + t, sizeX + f - 1); xSearch++) {

				++count;

				if (xSearch == x && ySearch == y && zSearch == z) continue;

				float d = 0;
				// Iterating over the local window of a voxel
				for (int lz = -f; lz <= f; lz++)
					for (int ly = -f; ly <= f; ly++)
						for (int lx = -f; lx <= f; lx++) {

							const int searchIndex = (zSearch + lz) * (sizeY + 2 * f) * (sizeX + 2 * f) +
								(xSearch + lx) * (sizeY + 2 * f) + (ySearch + ly),

								targetIndex = (z + lz) * (sizeY + 2 * f) * (sizeX + 2 * f) +
								(x + lx) * (sizeY + 2 * f) + (y + ly);


							d += powf(fabs(noisy[searchIndex] - noisy[targetIndex]), weight);
						}

				minimum = MIN(minimum, d);

				X += d;

				Y += noisy[zSearch * (sizeY + 2 * f) * (sizeX + 2 * f) +
					xSearch * (sizeY + 2 * f) + ySearch];

				denoisedImage[denoiseIndex] += d * noisy[zSearch * (sizeY + 2 * f) * (sizeX + 2 * f) +
					xSearch * (sizeY + 2 * f) + ySearch];
			}

	X += minimum;

	Y += noisy[z * (sizeY + 2 * f) * (sizeX + 2 * f) + x * (sizeY + 2 * f) + y];

	denoisedImage[denoiseIndex] += minimum * noisy[z * (sizeY + 2 * f) * (sizeX + 2 * f) + x * (sizeY + 2 * f) + y];

	denoisedImage[denoiseIndex] = (X * Y - denoisedImage[denoiseIndex]) / (count * X);

}



void launchParallelNLM(float *noisyData, float *kernel, float *denoisedImage, int f, int t, float h,
	int sizeX, int sizeY, int sizeZ, WeightingFunction weight) {

	float *d_noisyData, *d_kernel, *d_denoisedImage;
	dim3 blocks((sizeX + 7) / 8, (sizeY + 7) / 8, (sizeZ + 3) / 4), threads(8, 8, 4);
	cudaEvent_t start, end;

	cudaEventCreate(&start);
	cudaEventCreate(&end);


	cudaMalloc((void**)&d_noisyData, (sizeX + 2 * f) * (sizeY + 2 * f) * (sizeZ + 2 * f) * sizeof(float));
	cudaMalloc((void**)&d_kernel, (2 * f + 1) * (2 * f + 1) * (2 * f + 1) * sizeof(float));
	cudaMalloc((void**)&d_denoisedImage, sizeX * sizeY * sizeZ * sizeof(float));


	cudaMemcpy((void*)d_noisyData, (void*)noisyData, (sizeX + 2 * f) * (sizeY + 2 * f) * (sizeZ + 2 * f) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_kernel, (void*)kernel, (2 * f + 1) * (2 * f + 1) * (2 * f + 1) * sizeof(float), cudaMemcpyHostToDevice);


	switch (weight)
	{
	case GAUSSIAN_WEIGHTING:
		cudaEventRecord(start);
		parallelGaussianNLM << <blocks, threads >> > (d_noisyData, d_kernel, d_denoisedImage, f, t, h, sizeX, sizeY, sizeZ);
		cudaEventRecord(end);
		break;

	case L1_NORM:
	case L2_NORM:
	case L3_NORM:
	case L4_NORM:
		cudaEventRecord(start);
		parallelLkNormNLM << <blocks, threads >> > (d_noisyData, d_denoisedImage, f, t, sizeX, sizeY, sizeZ, weight);
		cudaEventRecord(end);
		break;
	}

	cudaMemcpy(denoisedImage, d_denoisedImage, sizeX * sizeY * sizeZ * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(end);


	cudaEventElapsedTime(&h, start, end);

	printf("\nParallel execution time L%d-NORM = %f seconds", weight, h / 1000);


	cudaCheckError();


	cudaFree(d_noisyData);
	cudaFree(d_kernel);
	cudaFree(d_denoisedImage);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

