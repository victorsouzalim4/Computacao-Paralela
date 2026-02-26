

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int sieveOfEratosthenes(int n)
{
   // Create a boolean array "prime[0..n]" and initialize
   // all entries it as true. A value in prime[i] will
   // finally be false if i is Not a prime, else true.
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);

   memset(prime, true,(n+1)*sizeof(bool));

   #pragma omp parallel 
   {
      int tid = omp_get_thread_num();
      #pragma omp for
      for (int p=2; p <= sqrt_n; p++){
         //printf("Thread %d\n", tid);
         // If prime[p] is not changed, then it is a prime
         if (prime[p] == true)
         {
            // Update all multiples of p
            for (int i=p*2; i<=n; i += p)
            prime[i] = false;
         }
      }
   }
   

    // count prime numbers
   #pragma omp parallel reduction( + : primes )
   {
      #pragma omp for
      for (int p=2; p<=n; p++)
         if (prime[p])
            primes++;
   }

   return(primes);

}

int main()
{
   int n = 100000000;
   printf("%d\n",sieveOfEratosthenes(n));
   return 0;
} 