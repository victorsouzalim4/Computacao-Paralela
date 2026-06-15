#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static double sum_seq(double *a, int width)
{
    double sum = 0.0;

    for (int i = 0; i < width; i++)
    {
        sum += a[i];
    }

    return sum;
}

static double sum_omp_multicore(double *a, int width)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < width; i++)
    {
        sum += a[i];
    }

    return sum;
}

static double sum_omp_gpu(double *a, int width)
{
    double sum = 0.0;

#pragma omp target teams distribute parallel for reduction(+ : sum) map(to : a[0 : width]) map(tofrom : sum)
    for (int i = 0; i < width; i++)
    {
        sum += a[i];
    }

    return sum;
}

static void run_test(const char *name, double (*fn)(double *, int), double *a, int width)
{
    double start = omp_get_wtime();
    double sum = fn(a, width);
    double end = omp_get_wtime();

    printf("%s\n", name);
    printf("tempo: %.6f segundos\n", end - start);
    printf("sum: %.6f\n\n", sum);
}

int main()
{
    int width = 40000000;
    size_t size = (size_t)width * sizeof(double);
    double *a = (double *)malloc(size);

    if (a == NULL)
    {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        return 1;
    }

    for (int i = 0; i < width; i++)
    {
        a[i] = i;
    }

    run_test("Sequencial", sum_seq, a, width);
    run_test("OpenMP multicore", sum_omp_multicore, a, width);
    run_test("OpenMP GPU target", sum_omp_gpu, a, width);

    free(a);

    return 0;
}
