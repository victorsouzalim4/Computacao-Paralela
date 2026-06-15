/*
 * Atividade 23 - Multiplicacao de matrizes com CUDA
 *
 * Compilacao:
 * nvcc -O3 mm_cuda.cu -o mm
 *
 * Execucao:
 * ./mm
 *
 * Metricas de GPU com ncu:
 * sudo env OMP_TARGET_OFFLOAD=MANDATORY ncu --target-processes all --replay-mode kernel \
 *     --metrics smsp__warps_launched.sum,smsp__thread_inst_executed_per_inst_executed.ratio ./mm
 *
 * Tempos de execucao para width = 2000, todas as versoes com -O3:
 * - Sequencial: 18.639014 segundos
 * - Paralela multicore OpenMP: 3.729357 segundos
 * - Melhor GPU OpenMP target: 4.364577 segundos
 *   warps_launched: 576
 *   warp_execution_efficiency: 100% aprox. (ncu ratio: 32.00)
 * - CUDA: 0.284670 segundos
 *   smsp__warps_launched.sum: 125000
 *   smsp__thread_inst_executed_per_inst_executed.ratio: 32
 */

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

static void check_cuda(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void mm_kernel(const double *a, const double *b, double *c, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        double sum = 0.0;

        for (int k = 0; k < width; k++)
        {
            double x = a[row * width + k];
            double y = b[k * width + col];
            sum += x * y;
        }

        c[row * width + col] = sum;
    }
}

static void init_matrices(double *a, double *b, double *c, int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            a[i * width + j] = (double)i;
            b[i * width + j] = (double)j;
            c[i * width + j] = 0.0;
        }
    }
}

static double checksum(const double *c, int width)
{
    double total = 0.0;
    int size = width * width;

    for (int i = 0; i < size; i++)
    {
        total += c[i];
    }

    return total;
}

static double expected_checksum(int width)
{
    double sum = ((double)width * (double)(width - 1)) / 2.0;
    return (double)width * sum * sum;
}

int main(int argc, char **argv)
{
    int width = 2000;

    if (argc > 1)
    {
        width = atoi(argv[1]);
    }

    if (width <= 0)
    {
        fprintf(stderr, "O tamanho da matriz deve ser positivo.\n");
        return EXIT_FAILURE;
    }

    int size = width * width;
    size_t bytes = (size_t)size * sizeof(double);

    double *a = (double *)malloc(bytes);
    double *b = (double *)malloc(bytes);
    double *c = (double *)malloc(bytes);

    if (a == NULL || b == NULL || c == NULL)
    {
        fprintf(stderr, "Erro ao alocar matrizes na CPU.\n");
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    init_matrices(a, b, c, width);

    double *d_a = NULL;
    double *d_b = NULL;
    double *d_c = NULL;

    check_cuda(cudaMalloc((void **)&d_a, bytes), "Erro ao alocar d_a");
    check_cuda(cudaMalloc((void **)&d_b, bytes), "Erro ao alocar d_b");
    check_cuda(cudaMalloc((void **)&d_c, bytes), "Erro ao alocar d_c");

    check_cuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "Erro ao copiar a para GPU");
    check_cuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "Erro ao copiar b para GPU");

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (width + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "Erro ao criar evento start");
    check_cuda(cudaEventCreate(&stop), "Erro ao criar evento stop");

    check_cuda(cudaEventRecord(start), "Erro ao iniciar medicao");
    mm_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
    check_cuda(cudaGetLastError(), "Erro ao lancar kernel");
    check_cuda(cudaEventRecord(stop), "Erro ao finalizar medicao");
    check_cuda(cudaEventSynchronize(stop), "Erro ao sincronizar kernel");

    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop), "Erro ao calcular tempo");

    check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "Erro ao copiar c para CPU");

    double result = checksum(c, width);
    double expected = expected_checksum(width);
    double tolerance = fabs(expected) * 1.0e-9;

    printf("width: %d\n", width);
    printf("tempo_cuda_kernel: %.6f segundos\n", (double)kernel_ms / 1000.0);
    printf("checksum: %.6f\n", result);
    printf("checksum_esperado: %.6f\n", expected);

    if (fabs(result - expected) > tolerance)
    {
        fprintf(stderr, "Erro: checksum difere do valor esperado.\n");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
