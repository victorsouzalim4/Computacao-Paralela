/*
 * Atividade 20 - Crivo de Eratostenes com OpenMP para GPU
 *
 *
 * Tempos de execucao para n = 100000000:
 * - Sequencial (-O3): 0.585019 segundos
 * - Paralela multicore OpenMP (-O3 -fopenmp): 2.244513 segundos
 * - Paralela GPU OpenMP target (-O3 -fopenmp): 2.117445 segundos
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *alloc_sieve(int n)
{
    unsigned char *prime = (unsigned char *)malloc((size_t)(n + 1));

    if (prime == NULL)
    {
        fprintf(stderr, "Erro ao alocar memoria para n = %d.\n", n);
        return NULL;
    }

    memset(prime, 1, (size_t)(n + 1));
    prime[0] = 0;

    if (n >= 1)
    {
        prime[1] = 0;
    }

    return prime;
}

static int integer_sqrt(int n)
{
    int r = 0;

    while ((long long)(r + 1) * (r + 1) <= n)
    {
        r++;
    }

    return r;
}

static int build_base_primes(int limit, int **base_primes_out)
{
    int count = 0;
    unsigned char *base = alloc_sieve(limit);
    int *base_primes = NULL;

    if (base == NULL)
    {
        return -1;
    }

    for (int p = 2; (long long)p * p <= limit; p++)
    {
        if (base[p])
        {
            for (int i = p * p; i <= limit; i += p)
            {
                base[i] = 0;
            }
        }
    }

    for (int p = 2; p <= limit; p++)
    {
        if (base[p])
        {
            count++;
        }
    }

    if (count == 0)
    {
        free(base);
        *base_primes_out = NULL;
        return 0;
    }

    base_primes = (int *)malloc((size_t)count * sizeof(int));
    if (base_primes == NULL)
    {
        fprintf(stderr, "Erro ao alocar lista de primos base.\n");
        free(base);
        return -1;
    }

    for (int p = 2, j = 0; p <= limit; p++)
    {
        if (base[p])
        {
            base_primes[j++] = p;
        }
    }

    free(base);
    *base_primes_out = base_primes;
    return count;
}

static int sieve_sequential(int n)
{
    int primes = 0;
    unsigned char *prime = alloc_sieve(n);

    if (prime == NULL)
    {
        return -1;
    }

    for (int p = 2; (long long)p * p <= n; p++)
    {
        if (prime[p])
        {
            for (int i = p * p; i <= n; i += p)
            {
                prime[i] = 0;
            }
        }
    }

    for (int p = 2; p <= n; p++)
    {
        primes += prime[p] != 0;
    }

    free(prime);
    return primes;
}

static int sieve_multicore_openmp(int n)
{
    int primes = 0;
    int limit = integer_sqrt(n);
    int *base_primes = NULL;
    int base_count = build_base_primes(limit, &base_primes);
    unsigned char *prime = alloc_sieve(n);

    if (base_count < 0 || prime == NULL)
    {
        free(base_primes);
        free(prime);
        return -1;
    }

    for (int j = 0; j < base_count; j++)
    {
        int p = base_primes[j];

#pragma omp parallel for schedule(static)
        for (int i = p * p; i <= n; i += p)
        {
            prime[i] = 0;
        }
    }

#pragma omp parallel for reduction(+ : primes) schedule(static)
    for (int p = 2; p <= n; p++)
    {
        primes += prime[p] != 0;
    }

    free(base_primes);
    free(prime);
    return primes;
}

static int sieve_gpu_openmp(int n)
{
    int primes = 0;
    int limit = integer_sqrt(n);
    int n_items = n + 1;
    int *base_primes = NULL;
    int base_count = build_base_primes(limit, &base_primes);
    unsigned char *prime = alloc_sieve(n);

    if (base_count < 0 || prime == NULL)
    {
        free(base_primes);
        free(prime);
        return -1;
    }

#pragma omp target data map(tofrom : prime[0 : n_items])
    {
        for (int j = 0; j < base_count; j++)
        {
            int p = base_primes[j];

#pragma omp target teams distribute parallel for
            for (int i = p * p; i <= n; i += p)
            {
                prime[i] = 0;
            }
        }

#pragma omp target teams distribute parallel for reduction(+ : primes) map(tofrom : primes)
        for (int p = 2; p <= n; p++)
        {
            primes += prime[p] != 0;
        }
    }

    free(base_primes);
    free(prime);
    return primes;
}

static void run_and_print(const char *label, int (*sieve)(int), int n)
{
    double start = omp_get_wtime();
    int primes = sieve(n);
    double end = omp_get_wtime();

    if (primes >= 0)
    {
        printf("%s: %d primos em %.6f segundos\n", label, primes, end - start);
    }
}

int main(int argc, char **argv)
{
    int n = 100000000;

    if (argc > 1)
    {
        n = atoi(argv[1]);
    }

    if (n < 2)
    {
        printf("0\n");
        return 0;
    }

    if (argc > 2 && strcmp(argv[2], "bench") == 0)
    {
        run_and_print("Sequencial", sieve_sequential, n);
        run_and_print("Multicore OpenMP", sieve_multicore_openmp, n);
        run_and_print("GPU OpenMP target", sieve_gpu_openmp, n);
        return 0;
    }

    printf("%d\n", sieve_gpu_openmp(n));
    return 0;
}
