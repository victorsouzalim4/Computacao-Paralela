#include <stdio.h>
#include <stdlib.h>

__global__ void sum_cuda(double *a, double *s, int width)
{
    int t = threadIdx.x;
    int b = blockIdx.x * blockDim.x;
    int lane = t % warpSize;
    int warp = t / warpSize;
    int warps_per_block = blockDim.x / warpSize;

    double sum = 0.0;

    if (b + t < width)
    {
        sum = a[b + t];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0)
    {
        s[blockIdx.x * warps_per_block + warp] = sum;
    }
}

int main()
{
    int width = 40000000;
    int size = width * sizeof(double);

    int block_size = 1024;
    int num_blocks = (width - 1) / block_size + 1;
    int warps_per_block = block_size / 32;
    int num_partials = num_blocks * warps_per_block;
    int s_size = num_partials * sizeof(double);

    double *a = (double *)malloc(size);
    double *s = (double *)malloc(s_size);

    if (a == NULL || s == NULL)
    {
        fprintf(stderr, "Erro ao alocar memoria no host.\n");
        free(a);
        free(s);
        return 1;
    }

    for (int i = 0; i < width; i++)
    {
        a[i] = i;
    }

    double *d_a;
    double *d_s;
    cudaEvent_t htd_start;
    cudaEvent_t htd_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    float htd_ms = 0.0f;
    float kernel_ms = 0.0f;

    cudaEventCreate(&htd_start);
    cudaEventCreate(&htd_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaMalloc((void **)&d_a, size);
    cudaEventRecord(htd_start);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaEventRecord(htd_stop);
    cudaEventSynchronize(htd_stop);
    cudaEventElapsedTime(&htd_ms, htd_start, htd_stop);

    cudaMalloc((void **)&d_s, s_size);
    cudaMemset(d_s, 0, s_size);

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(block_size, 1, 1);

    cudaEventRecord(kernel_start);
    sum_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

    cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);

    for (int i = 1; i < num_partials; i++)
    {
        s[0] += s[i];
    }

    printf("\nSum = %f\n", s[0]);
    printf("[CUDA memcpy HtoD] = %.6f ms\n", htd_ms);
    printf("sum_cuda(double*, double*, int) = %.6f ms\n", kernel_ms);

    cudaEventDestroy(htd_start);
    cudaEventDestroy(htd_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaFree(d_a);
    cudaFree(d_s);
    free(a);
    free(s);

    return 0;
}
