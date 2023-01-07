
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

__global__ void MaxPooling(unsigned char *imageRGBA, int width, int height, unsigned char *NewImageData)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width / 2 || y >= height / 2)
        return; // check if the thread is within the bounds of the image

    Pixel *ptrPixela = (Pixel *)&NewImageData[y * (width/2) * 4 + 2 * x];
    // Initialize max values to the first pixel in the 2x2 region
    Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
    unsigned char MAXR = 0;
    unsigned char MAXG = 0;
    unsigned char MAXB = 0;

    // Loop through the 2x2 region and find the max values for each color channel
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            //int currIndex = ((y+i)*width + (x+j)) * 4;
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

    // Set the pixel in the new image to the max values
    ptrPixela->r = MAXR;
    ptrPixela->g = MAXG;
    ptrPixela->b = MAXB;
    ptrPixela->a = 255; // alpha channel is always 255
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

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height, unsigned char *NewImageData)
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
__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA, unsigned char *NewImage, int width, int height )
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


  if(y < height && x < width)
    {
            float sum = 0;
            for (int i = 0; i < Ykernel; i++) {
              for (int j = 0; j < Xkernel; j++) {
                    int ky = i - Ykernel / 2;
                    int kx = j - Xkernel / 2;

                    int imy = y + ky;
                    int imx = x + kx;

                    if (imy >= 0 && imy < height && imx >= 0 && imx < width) {
                      // Convolve the kernel with the image
                      Pixel* ptrPixel = (Pixel*)&imageRGBA[imy * width * 4 + 4 * imx];
                      char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
                      sum1 += (pixelValue * kernel[i][j]);
                      sum2 += (pixelValue * kernel[i][j]);
                      sum3 += (pixelValue * kernel[i][j]);
                    }
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
    unsigned char* NewImageData = (unsigned char *)malloc(width * height * 4);
    unsigned char* NewImageDataconv = (unsigned char *)malloc(width * height * 4);
    unsigned char* OutputImage = (unsigned char *)malloc(width * height * 4); 
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
    //MaxPooling(imageData, width, height, NewImageData);
    MinPooling(imageData, width, height, NewImageData);
    //ConvertImageToGrayCpu(imageData, width, height, NewImageDataconv);
    printf(" DONE \r\n");

    
    
    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpu = nullptr;
    unsigned char* ptrImageOutGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMalloc(&ptrImageOutGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrImageOutGpu, OutputImage, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");
  
    // Process image on gpu
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    //ConvertImageToGrayGpu<<<gridSize, blockSize>>>(ptrImageDataGpu, ptrImageOutGpu , height, width);
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(OutputImage, ptrImageOutGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");
 
    // Build output filename
    const char * fileNameOut= "test.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, (width/2),(height/2), 4, NewImageData, 4 * (width/2));
    //stbi_write_png(fileNameOut, width-2, height-2, 4, OutputImage,  4*width);

    printf("DONE\r\n");
 
    // Free memory
    cudaFree(ptrImageDataGpu);
    cudaFree(ptrImageOutGpu);
    stbi_image_free(imageData);
    
}