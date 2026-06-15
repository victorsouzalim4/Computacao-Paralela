/*
 * Atividade 22 - Multiplicacao de matrizes com OpenMP para multicore e GPU
 *
 * Compilacao usada para todas as versoes:
 * gcc-8 -O3 -fopenmp mm_openmp_gpu.c -o mm
 * No ambiente WSL/RTX 4060, as versoes GPU foram compiladas com:
 * gcc -O3 -fopenmp -fcf-protection=none -fno-stack-protector \
 *     -foffload=nvptx-none mm_openmp_gpu.c -o mm
 *
 * Para medir cada versao, descomente apenas uma das diretivas indicadas dentro
 * da funcao mm(), compile novamente com -O3 -fopenmp e execute ./mm.
 *
 * Tempos de execucao para width = 2000:
 * - Sequencial (-O3): 18.639014 segundos
 * - Paralela multicore OpenMP (-O3 -fopenmp): 3.729357 segundos
 * - GPU OpenMP target teams distribute (-O3 -fopenmp): 8.508500 segundos
 *   warps_launched: 768
 *   warp_execution_efficiency: 100% aprox. (ncu ratio: 32.00)
 * - GPU OpenMP target teams distribute parallel for (-O3 -fopenmp): 4.854241 segundos
 *   warps_launched: 576
 *   warp_execution_efficiency: 100% aprox. (ncu ratio: 32.00)
 * - GPU OpenMP target teams distribute parallel for simd (-O3 -fopenmp): 4.364577 segundos
 *   warps_launched: 576
 *   warp_execution_efficiency: 100% aprox. (ncu ratio: 32.00)
 *
 * Metricas de GPU:
 * nvprof --events warps_launched --metrics warp_execution_efficiency ./mm
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void mm(double *a, double *b, double *c, int width)
{
    int size = width * width;

    /* Descomente apenas uma diretiva por compilacao. */
    /* Multicore: */
    // #pragma omp parallel for collapse(2)

    /* GPU - distribute: */
    // #pragma omp target teams distribute collapse(2) map(to : a[0 : size], b[0 : size]) map(from : c[0 : size])

    /* GPU - distribute parallel for: */
    // #pragma omp target teams distribute parallel for collapse(2) map(to : a[0 : size], b[0 : size]) map(from : c[0 : size])

    /* GPU - distribute parallel for simd: */
    // #pragma omp target teams distribute parallel for simd collapse(2) map(to : a[0 : size], b[0 : size]) map(from : c[0 : size])
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < width; k++)
            {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }

            c[i * width + j] = sum;
        }
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

static double checksum(double *c, int width)
{
    double total = 0.0;
    int size = width * width;

    for (int i = 0; i < size; i++)
    {
        total += c[i];
    }

    return total;
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
        return 1;
    }

    int size = width * width;
    double *a = (double *)malloc((size_t)size * sizeof(double));
    double *b = (double *)malloc((size_t)size * sizeof(double));
    double *c = (double *)malloc((size_t)size * sizeof(double));

    if (a == NULL || b == NULL || c == NULL)
    {
        fprintf(stderr, "Erro ao alocar matrizes.\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    init_matrices(a, b, c, width);

    double start = omp_get_wtime();
    mm(a, b, c, width);
    double end = omp_get_wtime();

    printf("width: %d\n", width);
    printf("tempo: %.6f segundos\n", end - start);
    printf("checksum: %.6f\n", checksum(c, width));

    free(a);
    free(b);
    free(c);

    return 0;
}
