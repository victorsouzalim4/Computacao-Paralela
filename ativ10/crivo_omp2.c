#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int sieve_of_eratosthenes(int n)
{
    int primes = 0;
    int sqrt_n = (int)sqrt((double)n);
    bool *prime = (bool *)malloc((size_t)(n + 1) * sizeof(bool));

    if (prime == NULL) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        return -1;
    }

    memset(prime, true, (size_t)(n + 1) * sizeof(bool));
    prime[0] = false;
    prime[1] = false;

    omp_set_num_threads(2);

#pragma omp parallel
    {
        for (int p = 2; p <= sqrt_n; p++) {
            if (prime[p]) {
#pragma omp for
                for (int i = p * 2; i <= n; i += p) {
                    prime[i] = false;
                }
            }
        }
    }

#pragma omp parallel for reduction(+ : primes)
    for (int p = 2; p <= n; p++) {
        if (prime[p]) {
            primes++;
        }
    }

    free(prime);
    return primes;
}

int main(int argc, char **argv)
{
    int n = 100000000;

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    printf("%d\n", sieve_of_eratosthenes(n));
    return 0;
}
