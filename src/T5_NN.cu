#include <iostream>
#include <math.h>
#include <stdio.h>
// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
// Include associated header file.
#include "../include/T5_NN.cuh"

__global__ void testGpu(float* x, float* A_row1, float* A_row2, float* A_row3, float* b)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row == 0)
    {
        // printf("%d %d A_row1[%d] %f x[%d] %f %f b[row]:%f\n", row, col, col, A_row1[col], col, x[col], A_row1[col]*x[col], b[row]);
        b[row*44 + col] = A_row1[col] * x[col];
    }
    if (row == 1)
    {
        // printf("%d %d A_row2[%d] %f x[%d] %f %f b[row]:%f\n", row, col, col, A_row2[col], col, x[col], A_row2[col]*x[col], b[row]);
        b[row*44 + col] = A_row2[col] * x[col];
    }
    if (row == 2)
    {
        // printf("%d %d A_row3[%d] %f x[%d] %f %f b[row]:%f\n", row, col, col, A_row3[col], col, x[col], A_row3[col]*x[col], b[row]);
        b[row*44 + col] = A_row3[col] * x[col];
    }
    return;
}

void test()
{
    // To perform Ax = b
    // where A = 3 x 44 matrix, x = 44 x 1 vector
    // input values
    float cpu_host_x[44] = { 1 , 2 , 3 , 4 , 5  , 6  , 7  , 8  , 9  , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 };
    float cpu_host_A_row1[44] = { 1 , 2 , 3 , 4 , 5  , 6  , 7  , 8  , 9  , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 };
    float cpu_host_A_row2[44] = { 3 , 4 , 5 , 6 , 7  , 8  , 9  , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 };
    float cpu_host_A_row3[44] = { 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 };
    float cpu_host_b[132]; // place to hold the result

    // Initialize x and allocate device memory for it
    float* gpu_device_x;
    float* gpu_device_A_row1;
    float* gpu_device_A_row2;
    float* gpu_device_A_row3;
    float* gpu_device_b; // memory space to hold results

    cudaMalloc(&gpu_device_x, 44*sizeof(float));
    cudaMalloc(&gpu_device_A_row1, 44*sizeof(float));
    cudaMalloc(&gpu_device_A_row2, 44*sizeof(float));
    cudaMalloc(&gpu_device_A_row3, 44*sizeof(float));
    cudaMalloc(&gpu_device_b, 132*sizeof(float));

    // Asynchronous copy
    cudaMemcpyAsync(gpu_device_x, cpu_host_x, 44*sizeof(float), cudaMemcpyHostToDevice, /*stream=*/0);
    cudaMemcpyAsync(gpu_device_A_row1, cpu_host_A_row1, 44*sizeof(float), cudaMemcpyHostToDevice, /*stream=*/0);
    cudaMemcpyAsync(gpu_device_A_row2, cpu_host_A_row2, 44*sizeof(float), cudaMemcpyHostToDevice, /*stream=*/0);
    cudaMemcpyAsync(gpu_device_A_row3, cpu_host_A_row3, 44*sizeof(float), cudaMemcpyHostToDevice, /*stream=*/0);
    cudaMemcpyAsync(gpu_device_b, cpu_host_b, 132*sizeof(float), cudaMemcpyHostToDevice, /*stream=*/0);

    // Synchronizing
    cudaStreamSynchronize(/*stream=*/0);

    dim3 blocksPerGrid(3, 1, 1);
    dim3 threadsPerBlock(44, 1, 1);

    // A                 times      x
    // <------------->              ^
    // <------------->              |
    // <------------->              V
    // three blocks
    // each with 44 threads

    testGpu<<<blocksPerGrid, threadsPerBlock>>>(gpu_device_x, gpu_device_A_row1, gpu_device_A_row2, gpu_device_A_row3, gpu_device_b);

    cudaDeviceSynchronize();
    
    cudaMemcpyAsync(cpu_host_b, gpu_device_b, 132*sizeof(float), cudaMemcpyDeviceToHost, /*stream=*/0);

    // Synchronizing
    cudaStreamSynchronize(/*stream=*/0);

    float result[3] = {0, 0, 0};
    for (int i = 0; i < 44; ++i)
    {
        result[0] += cpu_host_b[44 * 0 + i];
        result[1] += cpu_host_b[44 * 1 + i];
        result[2] += cpu_host_b[44 * 2 + i];
    }

    std::cout <<  " result[0]: " << result[0] <<  std::endl;
    std::cout <<  " result[1]: " << result[1] <<  std::endl;
    std::cout <<  " result[2]: " << result[2] <<  std::endl;

}
