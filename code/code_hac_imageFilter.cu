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


 

 __global__  void 2Dconvolution(float *ImageData, int matrix, int width, int height)
 {  
    /*
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    */

    //each block is assigned to a row of an image, iy integer index of y
    int iy = blockIdx.x + (MATRIX - 1)/2;
    //each thread is assigned to a pixel of a row, ix integer index of x
    int ix = threadIdx.x + (MATRIX - 1)/2;

    

 }

int main(int argc, char** argv)
{
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
    2Dconvolution<<gridSize,blockSize>>(ptrImageDataGpu, MATRIX, width, height );
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");
 
    // Build output filename
    switch argc:
    case 1


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

