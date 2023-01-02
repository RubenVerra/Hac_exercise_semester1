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

void calculate_kernel(int kernel_size, float sigma, float *kernel){

  int Nk2 = kernel_size*kernel_size;
  float x,y, center;

  center = (kernel_size-1)/2.0;
  
  for (int i = 0; i<Nk2; i++){
    x = (float)(i%kernel_size)-center;
    y =(float)(i/kernel_size)-center;
    kernel[i] = -(1.0/pi*pow(sigma,4))*(1.0 - 0.5*(x*x+y*y)/(sigma*sigma))*exp(-0.5*(x*x+y*y)/(sigma*sigma));
  }

}
 

 __global__ void Convolution(float *ImageData, float *kernel, float *NewImage, int matrix, int width, int height)
 {  
    //each block is assigned to a row of an image, iy integer index of y
    //each thread is assigned to a pixel of a row, ix integer index of x
    int iy = blockIdx.x + (MATRIX - 1)/2;
    int ix = threadIdx.x + (MATRIX - 1)/2;

    int center = (MATRIX -1)/2;
    int idx = iy*width +ix;
    
    int tid = threadIdx.x;
    int K2 = MATRIX*MATRIX;
    __shared__ float sdata[9];

    if (tid<K2)
    {
        sdata[tid] = kernel[tid];
    }
    __syncthreads();

    int ii, jj;
    float sum = 0.0;

    if (idx<width*height)
    {
        for (int ki = 0; ki<MATRIX; ki++)
            for (int kj = 0; kj<MATRIX; kj++){
             ii = kj + ix - center;
             jj = ki + iy - center;
            sum+=ImageData[jj*width+ii]*sdata[ki*MATRIX + kj];
        }
        NewImage[idx] = sum;
  }

 }

int main(int argc, char** argv)
{
    float *kernel = (float*)malloc(MATRIX*MATRIX*sizeof(float));  
    kernel_size = 3;
    
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
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
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
    calculate_kernel(kernel_size, sigma, kernel);

    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    float* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");
 
    // Process image on gpu
    float* NewImage = (float*)malloc(height*width*sizeof(float));
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    Convolution<<<gridSize,blockSize>>>(ptrImageDataGpu, kernel, NewImage, MATRIX, width, height );
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");
 
    // Build output filename



    const char * fileNameOut = "image1.png";


    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width, height, 4, NewImage, 4 * width);
    printf("DONE\r\n");
 
    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
}


