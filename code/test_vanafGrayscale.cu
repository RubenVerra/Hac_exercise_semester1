
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"



struct Pixel
{
 unsigned char r, g, b, a;
};

#define N	50000

void MaxPooling(unsigned char *imageRGBA, int width, int height, unsigned char *NewImageData)
{
    int t = 0;

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            Pixel *ptrPixela = (Pixel *)&NewImageData[t];
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char MAXR = 0;
            unsigned char MAXG = 0;
            unsigned char MAXB = 0;

            for (int c = 0; c < 4; c++)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                      if(MAXR < ptrPixel->r)
                    {
                      MAXR = ptrPixel->r;
                    }
                      if(MAXG < ptrPixel->g)
                    {
                      MAXG = ptrPixel->g;
                    }
                      if(MAXB < ptrPixel->b)
                    {
                      MAXB = ptrPixel->b;
                    }
                    }
                }

                ptrPixela->r = MAXR;
                ptrPixela->g = MAXG;
                ptrPixela->b = MAXB;
                ptrPixela->a = 255;
                t++;
            }
        }
    }
}

__global__ void MaxPoolingGPU(unsigned char *imageRGBA, int width, int height, unsigned char *NewImageData)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process valid elements of the input and output arrays
    if (x < width && y < height && x % 2 == 0 && y % 2 == 0)
    {
        int t = y * (width) + x*2;
        Pixel *ptrPixela = (Pixel *)&NewImageData[t];
        Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
        unsigned char MAXR = 0;
        unsigned char MAXG = 0;
        unsigned char MAXB = 0;

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if(MAXR < ptrPixel->r)
                {
                    MAXR = ptrPixel->r;
                }
                if(MAXG < ptrPixel->g)
                {
                    MAXG = ptrPixel->g;
                }
                if(MAXB < ptrPixel->b)
                {
                    MAXB = ptrPixel->b;
                }
            }
        }

        ptrPixela->r = MAXR;
        ptrPixela->g = MAXG;
        ptrPixela->b = MAXB;
        ptrPixela->a = 255;
    }
}


void MinPooling(unsigned char *imageRGBA, int width, int height, unsigned char *NewImageData)
{
    int t = 0;

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            Pixel *ptrPixela = (Pixel *)&NewImageData[t];
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char MINR = 255;
            unsigned char MING = 255;
            unsigned char MINB = 255;

            for (int c = 0; c < 4; c++)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                      if(MINR > ptrPixel->r)
                    {
                      MINR = ptrPixel->r;
                    }
                      if(MING > ptrPixel->g)
                    {
                      MING = ptrPixel->g;
                    }
                      if(MINB > ptrPixel->b)
                    {
                      MINB = ptrPixel->b;
                    }
                    }
                }

                ptrPixela->r = MINR;
                ptrPixela->g = MING;
                ptrPixela->b = MINB;
                ptrPixela->a = 255;
                t++;
            }
        }
    }
}

__global__ void MinPoolingGPU(unsigned char *imageRGBA, int width, int height, unsigned char *NewImageData)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process valid elements of the input and output arrays
    if (x < width && y < height && x % 2 == 0 && y % 2 == 0)
    {
        int t = y * (width) + x*2;
        Pixel *ptrPixela = (Pixel *)&NewImageData[t];
        Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
        unsigned char MAXR = 255;
        unsigned char MAXG = 255;
        unsigned char MAXB = 255;

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if(MAXR > ptrPixel->r)
                {
                    MAXR = ptrPixel->r;
                }
                if(MAXG > ptrPixel->g)
                {
                    MAXG = ptrPixel->g;
                }
                if(MAXB > ptrPixel->b)
                {
                    MAXB = ptrPixel->b;
                }
            }
        }

        ptrPixela->r = MAXR;
        ptrPixela->g = MAXG;
        ptrPixela->b = MAXB;
        ptrPixela->a = 255;
    }
}

void convolutionCPU(unsigned char* imageRGBA, int width, int height, unsigned char *NewImageData)
{
    const int Ykernel = 3;
    const int Xkernel = 3;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    float kernel[Ykernel][Xkernel] =
            {  
            {0, -1, 0},
            {-1, 8, -1},
            {0, -1, 0}
            };


    for (int y = 0; y < height - 2; y++)
    {
        for (int x = 0; x < width - 2; x++)
        {
            for(int i = 0; i < Ykernel; i++)
            {
                for(int j = 0; j <Xkernel; j++)
                {
                    Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
                    
                    char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
                    sum1 += (pixelValue * kernel[i][j]);
                    sum2 += (pixelValue * kernel[i][j]);
                    sum3 += (pixelValue * kernel[i][j]);
                    //printf("sum1 = %d\n ",sum1);
                }
            }
            Pixel* ptrPxl = (Pixel*)&NewImageData[y * width * 4 + 4 * x];
            ptrPxl->r = sum1;
            ptrPxl->g = sum2;
            ptrPxl->b = sum3;
            ptrPxl->a = 255;

            sum1 = 0;
            sum2 = 0;
            sum3 = 0;            
        }
    }
}
__global__ void convolutionGPU(unsigned char* imageRGBA, unsigned char *NewImage, int width, int height )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    const int Ykernel = 3;
    const int Xkernel = 3;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    float kernel[Ykernel][Xkernel] =
            {  
            {0, -1, 0},
            {-1, 8, -1},
            {0, -1, 0}
            };


  if(y < (height-2) && x < (width-2) )
    {
            float sum = 0;
            for (int i = 0; i < Ykernel; i++) {
              for (int j = 0; j < Xkernel; j++) {
                   // int ky = i - Ykernel / 2;
                   // int kx = j - Xkernel / 2;

                    //int imy = y + ky;
                    //int imx = x + kx;

                    //if (imy >= 0 && imy < height && imx >= 0 && imx < width) {
                      // Convolve the kernel with the image
                      //Pixel* ptrPixel = (Pixel*)&imageRGBA[imy * width * 4 + 4 * imx];

                      Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
                      char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
                      sum1 += (pixelValue * kernel[i][j]);
                      sum2 += (pixelValue * kernel[i][j]);
                      sum3 += (pixelValue * kernel[i][j]);
                   // }
              }
            }
                // Set the output value for the current pixel
            Pixel* ptrPixel = (Pixel*)&NewImage[y * width * 4 + 4 * x];
            ptrPixel->r = sum1;
            ptrPixel->g = sum2;
            ptrPixel->b = sum3;
            ptrPixel->a = 255;

            sum1 = 0;
            sum2 = 0;
            sum3 = 0; 
  }
}


int main(int argc, char** argv)
{
    //GPU CUDA events
    cudaEvent_t c_start, c_stop, max_start, max_stop, min_start, min_stop;
    	cudaEventCreate(&c_start);
	    cudaEventCreate(&c_stop);
      cudaEventCreate(&max_start);
	    cudaEventCreate(&max_stop);
      cudaEventCreate(&min_start);
	    cudaEventCreate(&min_stop);
    
    //CPU CUDA events
    cudaEvent_t c_startCPU, c_stopCPU, max_startCPU, max_stopCPU, min_startCPU, min_stopCPU;
    	cudaEventCreate(&c_startCPU);
	    cudaEventCreate(&c_stopCPU);
      cudaEventCreate(&max_startCPU);
	    cudaEventCreate(&max_stopCPU);
      cudaEventCreate(&min_startCPU);
	    cudaEventCreate(&min_stopCPU);
    
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

    //CPU output
    unsigned char* NewImageDataMaxPool = (unsigned char *)malloc(width * height * 4);
    unsigned char* NewImageDataMinPool = (unsigned char *)malloc(width * height * 4);
    unsigned char* NewImageDataconv = (unsigned char *)malloc(width * height * 4);

    //GPU output
    unsigned char* OutputImage = (unsigned char *)malloc(width * height * 4);
    unsigned char* OutputImageMaxPool = (unsigned char *)malloc(width * height * 4);
    unsigned char* OutputImageMinPool = (unsigned char *)malloc(width * height * 4); 


    
    
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
    printf("Processing image...\r\n");
//--------------------------------------------------------------------------------

    cudaEventRecord(c_startCPU, 0);

    convolutionCPU(imageData, width, height, NewImageDataconv);

    cudaEventRecord(c_stopCPU, 0);
	  cudaEventSynchronize(c_stopCPU);
	  float elapsedTimeCPU;
	  cudaEventElapsedTime(&elapsedTimeCPU, c_startCPU, c_stopCPU);
	  printf("Time to complete CPU 2Dconvolution: %3.1f ms\n\r", elapsedTimeCPU);

//--------------------------------------------------------------------------------

    cudaEventRecord(max_startCPU, 0);
    MaxPooling(imageData, width, height, NewImageDataMaxPool);

    cudaEventRecord(max_stopCPU, 0);
	  cudaEventSynchronize(max_stopCPU);
	  float elapsedTimeMaxCPU;
	  cudaEventElapsedTime(&elapsedTimeMaxCPU, max_startCPU, max_stopCPU);
	  printf("Time to complet CPU maxPooling : %3.1f ms\n\r", elapsedTimeMaxCPU);

//--------------------------------------------------------------------------------

    cudaEventRecord(min_startCPU, 0);

    MinPooling(imageData, width, height, NewImageDataMinPool);

    cudaEventRecord(min_stopCPU, 0);
	  cudaEventSynchronize(min_stopCPU);
	  float elapsedTimeMinCPU;
	  cudaEventElapsedTime(&elapsedTimeMinCPU, min_startCPU, min_stopCPU);
	  printf("Time to complete CPU minPooling: %3.1f ms\n\r", elapsedTimeMinCPU);

//--------------------------------------------------------------------------------
    printf(" DONE \r\n");

    cudaEventRecord(c_start, 0);
    
    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpu = nullptr;
    unsigned char* ptrImageOutGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMalloc(&ptrImageOutGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrImageOutGpu, OutputImage, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");
  
    // Process image on gpu (convolution)
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    convolutionGPU<<<gridSize, blockSize>>>(ptrImageDataGpu, ptrImageOutGpu , height, width);
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu (convolution)
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(OutputImage, ptrImageOutGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");

  	
	  cudaEventRecord(c_stop, 0);
	  cudaEventSynchronize(c_stop);
	  float elapsedTime;
	  cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
	  printf("Time to complete 2Dconvolution: %3.1f ms\n\r", elapsedTime);
    
    cudaEventRecord(max_start, 0);

    // Copy data to the gpu (MaxPooling)
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpuMx = nullptr;
    unsigned char* ptrImageOutGpuMx = nullptr;
    cudaMalloc(&ptrImageDataGpuMx, width * height * 4);
    cudaMalloc(&ptrImageOutGpuMx, width * height * 4);
    cudaMemcpy(ptrImageDataGpuMx, imageData, width * height * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrImageOutGpuMx, OutputImageMaxPool, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");


    // Process image on gpu (MaxPooling)
    printf("Running CUDA Kernel...\r\n"); 
    MaxPoolingGPU<<<gridSize, blockSize>>>(ptrImageDataGpuMx, width , height, ptrImageOutGpuMx);
    printf(" DONE \r\n" );

    // Copy data from the gpu (MaxPooling)
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpuMx, width * height * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(OutputImageMaxPool, ptrImageOutGpuMx, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");

    cudaEventRecord(max_stop, 0);
	  cudaEventSynchronize(max_stop);
	  float elapsedTimeMax;
	  cudaEventElapsedTime(&elapsedTimeMax, max_start, max_stop);
	  printf("Time to complet maxPooling : %3.1f ms\n\r", elapsedTimeMax);

    cudaEventRecord(min_start, 0);

    // Copy data to the gpu (MinPooling)
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpuMn = nullptr;
    unsigned char* ptrImageOutGpuMn = nullptr;
    cudaMalloc(&ptrImageDataGpuMn, width * height * 4);
    cudaMalloc(&ptrImageOutGpuMn, width * height * 4);
    cudaMemcpy(ptrImageDataGpuMn, imageData, width * height * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrImageOutGpuMn, OutputImageMinPool, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");

    // Process image on gpu (MinPooling)
    printf("Running CUDA Kernel...\r\n"); 
    MinPoolingGPU<<<gridSize, blockSize>>>(ptrImageDataGpuMx, width , height, ptrImageOutGpuMn);
    printf(" DONE \r\n" );

    // Copy data from the gpu (MinPooling)
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpuMn, width * height * 4, cudaMemcpyDeviceToHost);
<<<<<<< Updated upstream
    cudaMemcpy(OutputImageMinPool, ptrImageOutGpuMn, width * height * 4, cudaMemcpyDeviceToHost);
=======
    cudaMemcpy(OutputImageMaxPool, ptrImageOutGpuMn, width * height * 4, cudaMemcpyDeviceToHost);
>>>>>>> Stashed changes
    printf(" DONE \r\n");

    cudaEventRecord(min_stop, 0);
	  cudaEventSynchronize(min_stop);
	  float elapsedTimeMin;
	  cudaEventElapsedTime(&elapsedTimeMin, min_start, min_stop);
	  printf("Time to complete minPooling: %3.1f ms\n\r", elapsedTimeMin);
 
    // Build output filename
    const char * fileNameOut_a= "ConvCPU.png";
    const char * fileNameOut_b= "MaxPoolCPU.png";
    const char * fileNameOut_c= "MinPoolCPU.png";

    const char * fileNameOut_d= "ConvGPU.png";
    const char * fileNameOut_e= "MaxPoolGPU.png";
    const char * fileNameOut_f= "MinPoolGPU.png";


    // Write image back to disk
    printf("Writing png to disk...\r\n");

    //write out CPU
    stbi_write_png(fileNameOut_b, (width/2),(height/2), 4, NewImageDataMaxPool, 4 * (width/2));
    stbi_write_png(fileNameOut_c, (width/2),(height/2), 4, NewImageDataMinPool, 4 * (width/2));
    stbi_write_png(fileNameOut_a, width-2, height-2, 4, NewImageDataconv,  4*width);
    
    //write out GPU
    stbi_write_png(fileNameOut_d, width-2, height-2, 4, OutputImage,  4*width);
    stbi_write_png(fileNameOut_e, (width/2),(height/2), 4, OutputImageMaxPool, 4 * (width/2));
<<<<<<< Updated upstream
    stbi_write_png(fileNameOut_f, (width/2),(height/2), 4, OutputImageMinPool, 4 * (width/2));
=======
    stbi_write_png(fileNameOut_e, (width/2),(height/2), 4, OutputImageMinPool, 4 * (width/2));
>>>>>>> Stashed changes


    printf("DONE\r\n");
 
    // Free memory
    cudaFree(ptrImageDataGpu);
    cudaFree(ptrImageOutGpu);

    cudaFree(ptrImageDataGpuMx);
    cudaFree(ptrImageOutGpuMx);
    cudaFree(ptrImageDataGpuMn);

    stbi_image_free(imageData);
    
}