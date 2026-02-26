#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for(int i = 1; i <= 3; i++) 
        {
           printf("[PRINT1] T%d = %d \n",tid,i);
           printf("[PRINT2] T%d = %d \n",tid,i);
        }
    }
}