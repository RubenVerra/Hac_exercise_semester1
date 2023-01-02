#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "stb_image.h"
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define MATRIX  9
#define WMATRIX 3
#define HMATRIX 3


 

 __global__ void Convolution(float *ImageData, float *NewImage, int matrix, int width, int height)
 {  
    //each block is assigned to a row of an image, iy integer index of y
    //each thread is assigned to a pixel of a row, ix integer index of x
    int iy = blockIdx.x + (MATRIX - 1)/2;
    int ix = threadIdx.x + (MATRIX - 1)/2;

    int center = (MATRIX -1)/2;
    int idx = iy*width +ix;
    int sum = 0;

    int tid = threadIdx.x;
    int K2 = MATRIX*MATRIX;
    __shared__ float sdata[9];

    if (tid<K2)
    {
        sdata[tid] = kernel[tid];
    }
    __syncthreads();

    if (idx<width*height)
    {
        for (int ki = 0; ki<WMATRIX; ki++)
            for (int kj = 0; kj<HMATRIX; kj++){
            int ii = kj + ix - center;
            int jj = ki + iy - center;
            sum+=ImageData[jj*width+ii]*sdata[ki*MATRIX + kj];
        }
        NewImage[idx] = sum;
  }

 }

int main(int argc, char** argv)
{
    unsigned char* NewImage;

    for(int i = 1; i < argc; i++)
    // Check argument count
    if (argc < 2)
    {
        printf("Usage: im2gray <filename>\r\n");
        return -1;
    }
 
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char* imageData = stbi_load(argv[i], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        return -1;
    }
    printf(" DONE \r\n" );
 
 
    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32!\r\n");
        return -1;
    }
 
    
    // Process image on cpu
    /*
    printf("Processing image...\r\n");
    
    printf(" DONE \r\n");
    */

    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");
 
    // Process image on gpu
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    2Dconvolution<<gridSize,blockSize>>(ptrImageDataGpu, NewImage, MATRIX, width, height );
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");
 
    // Build output filename



    switch(i) {
  case 1:
    const char * fileNameOut = "image1.png";  
  case 2:
    const char * fileNameOut = "image2.png";
  case 3:
    const char * fileNameOut = "image3.png";
  default:

}

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width, height, 4, imageData, 4 * width);
    printf("DONE\r\n");
 
    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
}

