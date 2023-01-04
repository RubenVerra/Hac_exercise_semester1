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

void MaxPooling(unsigned char* imageRGBA, int width, int height, unsigned char *NewImageData )
{
    const int Ykernel = 2;
    const int Xkernel = 2;

    unsigned char red[2][2];
    unsigned char blue[2][2];
    unsigned char green[2][2];    
    
     int max1 = 0;
     int max2 = 0;
     int max3 = 0;

    for (int y = 0; y < height; y += 2)
    {
    
        for (int x = 0; x < width; x += 2)
        {
            for(int i = 0; i < Ykernel; i++)
            {
                for(int j = 0; j < Xkernel; j++)
                {
                    Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
                    red[i][j] = ptrPixel->r;
                    green[i][j] = ptrPixel->g;
                    blue[i][j] = ptrPixel->b;
                                   
                    //printf("sum1 = %d\n ",sum1);
                    
                    //printf("kernel = %d\n ",kernel[i][j]);
                }
            }
            int max1 = red[0][0];
            
            for(int r = 0; r < 2; r++)
            {
                for(int f = 0; f < 2; f++ )
                {
                    if(max1 < red[r][f])
                    {
                        max1 = red[r][f];
                    }
                }
            }

            int max2 = green[0][0];
            
            for(int r = 0; r < 2; r++)
            {
                for(int f = 0; f < 2; f++ )
                {
                    if(max2 < green[r][f])
                    {
                        max2 = green[r][f];
                    }
                }
            }

            int max3 = blue[0][0];
            
            for(int r = 0; r < 2; r++)
            {
                for(int f = 0; f < 2; f++ )
                {
                    if(max3 < blue[r][f])
                    {
                        max3 = blue[r][f];
                    }
                }
            }
            Pixel* ptrPixela = (Pixel*)&NewImageData[y * width * 4 + 4 * x];
            ptrPixela->r = max1;
            ptrPixela->g = max2;
            ptrPixela->b = max3;
            ptrPixela->a = 255;           
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
 /*
__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA, int width, int height, unsigned char *NewImage )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int kernelX = ;
    int kernelY = ;


  if(y < height && x < width)
    {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r/3 + ptrPixel->g / 3 + ptrPixel->b / 3);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
    }
}
*/

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
    unsigned char* OutputImage; 
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
    ConvertImageToGrayCpu(imageData, width, height, NewImageDataconv);
    //MaxPooling(imageData, width, height, NewImageData);
    printf(" DONE \r\n");

    
    
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
  //  ConvertImageToGrayGpu<<<gridSize, blockSize>>>(ptrImageDataGpu, height, width, OutputImage);
    printf(" DONE \r\n" ); 
 
    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");
 
    // Build output filename
    const char * fileNameOut= "test.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width-2, height-2, 4, NewImageData, 4 * width);
    //stbi_write_png(fileNameOut, width/2, height/2, 4, NewImageData, 4 * width/2);

    printf("DONE\r\n");
 
    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
    
}