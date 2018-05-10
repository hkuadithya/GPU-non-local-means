/*
=================================================
			Adithya H K Upadhya
			MS CS, Virginia Tech
			adithyau@vt.edu
			4/25/2018
=================================================
*/

#include <fstream>
#include <string>
#include <time.h>
#include "getopt.h"
#include "gpu_device_code.cuh"

using namespace std;


std::string path = "C:/Users/hkuad/source/repos/ProjectFiles/";

// Generates a 3D Gaussian Kernel
float* gaussianKernel(int f, float sigma) {

	int l = 2 * f + 1;

	float *kernel = new float[l * l * l], sum = 0;

	for (int i = -f; i <= f; i++)
		for (int j = -f; j <= f; j++)
			for (int k = -f; k <= f; k++) {

				int offset = (i + f) * l * l + (j + f) * l + (k + f);

				kernel[offset] = expf(-(i * i + j * j + k * k) / (2 * sigma * sigma));

				sum += kernel[offset];
			}

	for (int i = 0; i < l * l * l; i++) kernel[i] /= sum;

	return kernel;
}

// Writes the data to a binary file, output_gpu.bin or outpu_cpu.bin
void writeDenoisedImageData(string fileName, float* denoisedImage, int sizeX, int sizeY, int sizeZ) {

	ofstream out(path + fileName, ios::out | ios::binary);

	for (int z = 0; z < sizeZ; z++)
		for (int y = 0; y < sizeY; y++)
			for (int x = 0; x < sizeX; x++)
				out.write((char *)(denoisedImage + z * sizeY * sizeX + x * sizeY + y), sizeof(float));

	out.close();
}

// Reads the noisy image data from input binary file.
float* readNoisyImageData(int sizeX, int sizeY, int sizeZ) {

	ifstream in(path + "input.bin", ios::in | ios::binary);

	float *noisyImageData = new float[sizeX * sizeY * sizeZ];


	for (int z = 0; z < sizeZ; z++)
		for (int y = 0; y < sizeY; y++)
			for (int x = 0; x < sizeX; x++) {
				in.read((char *)(noisyImageData + z * sizeY * sizeX + x * sizeY + y), sizeof(float));
			}

	in.close();

	printf("\nImage data reading complete...");

	return noisyImageData;
}

// NLM serial execution is performed here.
float* nlmDenoisingSerial(float *__restrict noisy, float *__restrict kernel, float *__restrict denoisedImage,
	int f, int t, float h, int sizeX, int sizeY, int sizeZ) {

	printf("\nRunning Serial NL means...");
	// Iterate over all the interal Voxels of Noisy Image.
	for (int z = f; z < sizeZ + f; z++) {
		for (int x = f; x < sizeX + f; x++) {
			for (int y = f; y < sizeY + f; y++) {

				//z * sizeY * sizeX + x * sizeY + y
				int denoiseIndex = (z - f) * sizeY * sizeX + (x - f) * sizeY + (y - f);

				denoisedImage[denoiseIndex] = 0;

				float normalize = 0, maximum = 0;

				// Iterating over the Search Window of the voxel.
				// Boundary checks applied here through MAX & MIN macros.
				for (int zSearch = MAX(z - t, f); zSearch <= MIN(z + t, sizeZ + f - 1); zSearch++)
					for (int xSearch = MAX(x - t, f); xSearch <= MIN(x + t, sizeX + f - 1); xSearch++)
						for (int ySearch = MAX(y - t, f); ySearch <= MIN(y + t, sizeY + f - 1); ySearch++) {

							if (xSearch == x && ySearch == y && zSearch == z) continue;

							float d = 0;
							// Iterating over the local window of a voxel
							for (int lz = -f; lz <= f; lz++)
								for (int lx = -f; lx <= f; lx++)
									for (int ly = -f; ly <= f; ly++) {


										int searchIndex = (zSearch + lz) * (sizeY + 2 * f) * (sizeX + 2 * f) +
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
		}
	}

	return denoisedImage;
}

// Utility method for launching NLM serial algorithm.
void launchSerialNLM(float *noisyData, float * kernel, float *denoisedImage, int f, int t, float h, int sizeX, int sizeY, int sizeZ) {

	clock_t startTime = clock();

	nlmDenoisingSerial(noisyData, kernel, denoisedImage, f, t, h, sizeX, sizeY, sizeZ);

	printf("\nSerial execution time (t-%d) = %lf seconds", t, (clock() - startTime) / (double)CLOCKS_PER_SEC);

}


int main(int argc, char* argv[])
{

	int f = 2, t = 5, sizeX = 181, sizeY = 217, sizeZ = 51;

	float sigma = 1, h = 10;

	char option;

	/*	COMMAND LINE ARGUMENTS (These have already been set in the project)

		f	:	local window Radius
		t	:	search window Radius
		s	:	standard deviation of the Gaussian kernel
		h	:	standard deviation of the Gaussian Weighting function
		x	:	x-dimension of image
		y	:	y-dimension of image
		z	:	z-dimension of image
	*/

	while ((option = getopt(argc, argv, "f:t:s:h:x:y:z:")) != -1) {

		switch (option) {

		case 'f':
			f = atoi(optarg);
			break;
		case 't':
			t = atoi(optarg);
			break;
		case 's':
			sigma = atof(optarg);
			break;
		case 'h':
			h = atof(optarg);
			break;
		case 'x':
			sizeX = atoi(optarg);
			break;
		case 'y':
			sizeY = atoi(optarg);
			break;
		case 'z':
			sizeZ = atoi(optarg);
			break;
		default:
			printf("\nProgram terminated! Wrong Command Line Arguments specified...\n");
			exit(1);
		}
	}

	printf("\nf=%d t=%d sigma=%f h=%f sizeX=%d sizeY=%d sizeZ=%d\n", f, t, sigma, h, sizeX, sizeY, sizeZ);


	float *noisyData = readNoisyImageData(sizeX + 2 * f, sizeY + 2 * f, sizeZ + 2 * f),
		*kernel = gaussianKernel(f, sigma),
		*denoisedImage = new float[sizeX * sizeY * sizeZ];


	//-------------------	NLM	Serial	----------------------------


	launchSerialNLM(noisyData, kernel, denoisedImage, f, t, h, sizeX, sizeY, sizeZ);

	writeDenoisedImageData("output_cpu.bin", denoisedImage, sizeX, sizeY, sizeZ);


	//-------------------	NLM	Parallel ---------------------------



	launchParallelNLM(noisyData, kernel, denoisedImage, f, t, h, sizeX, sizeY, sizeZ, GAUSSIAN_WEIGHTING);

	writeDenoisedImageData("output_gpu.bin", denoisedImage, sizeX, sizeY, sizeZ);



	delete[] noisyData, kernel, denoisedImage;

	return 0;
}