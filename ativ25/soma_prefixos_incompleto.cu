/*
 * Atividade 25 - Soma de prefixos em CUDA
 *
 * Compilacao:
 * nvcc -O3 soma_prefixos_incompleto.cu -o soma_prefixos
 *
 * Execucao:
 * ./soma_prefixos
 *
 * Profiling neste ambiente com ncu:
 * sudo ncu --target-processes all --replay-mode kernel ./soma_prefixos
 *
 * Tempos para N = 65536:
 * - CPU sequencial: 0.014735 ms
 * - CUDA kernels: 214.268 ms
 *
 * Saida de verificacao:
 * - ultimo_prefixo_cpu: 294187
 * - ultimo_prefixo_gpu: 294187
 *
 * Duracao dos kernels pelo ncu:
 * - somaPrefixosBlocos: 5.34 us
 * - somaPrefixosBlocosPequeno: 3.10 us
 * - adicionaOffsets: 3.58 us
 *
 * Observacao: executando sob ncu, o tempo total medido pelos eventos CUDA subiu
 * para 446.243 ms por causa do overhead do profiler.
 *
 * Observacao: o nvprof foi substituido por ncu nas instalacoes CUDA mais novas.
 * Se nvprof estiver disponivel, execute:
 * nvprof ./soma_prefixos
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256

static void checkCuda(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        std::cerr << message << ": " << cudaGetErrorString(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void somaPrefixosCpu(const int *arr, int *somas, int tamanho)
{
    if (tamanho <= 0)
    {
        return;
    }

    somas[0] = arr[0];

    for (int i = 1; i < tamanho; i++)
    {
        somas[i] = somas[i - 1] + arr[i];
    }
}

__global__ void somaPrefixosBlocos(const int *v, int *v_somas, int *somas_blocos, int tamanho)
{
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (gid < tamanho) ? v[gid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int valor = 0;

        if (tid >= offset)
        {
            valor = temp[tid - offset];
        }

        __syncthreads();
        temp[tid] += valor;
        __syncthreads();
    }

    if (gid < tamanho)
    {
        v_somas[gid] = temp[tid];
    }

    if (tid == blockDim.x - 1)
    {
        somas_blocos[blockIdx.x] = temp[tid];
    }
}

__global__ void somaPrefixosBlocosPequeno(int *somas_blocos, int num_blocos)
{
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    temp[tid] = (tid < num_blocos) ? somas_blocos[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int valor = 0;

        if (tid >= offset)
        {
            valor = temp[tid - offset];
        }

        __syncthreads();
        temp[tid] += valor;
        __syncthreads();
    }

    if (tid < num_blocos)
    {
        somas_blocos[tid] = temp[tid];
    }
}

__global__ void adicionaOffsets(int *v_somas, const int *somas_blocos, int tamanho)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < tamanho)
    {
        v_somas[gid] += somas_blocos[blockIdx.x - 1];
    }
}

int main()
{
    const int N = 1 << 16;
    const size_t bytes = (size_t)N * sizeof(int);
    const int num_blocos = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t bytes_blocos = (size_t)num_blocos * sizeof(int);

    if (num_blocos > BLOCK_SIZE)
    {
        std::cerr << "Esta implementacao suporta ate " << BLOCK_SIZE
                  << " blocos. Reduza N ou aumente BLOCK_SIZE.\n";
        return EXIT_FAILURE;
    }

    std::vector<int> host_arr(N);
    std::vector<int> host_somas_cpu(N);
    std::vector<int> host_somas_gpu(N);

    std::srand(0);
    for (int i = 0; i < N; i++)
    {
        host_arr[i] = std::rand() % 10;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    somaPrefixosCpu(host_arr.data(), host_somas_cpu.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    int *device_arr = NULL;
    int *device_somas_prefixos = NULL;
    int *device_somas_blocos = NULL;

    checkCuda(cudaMalloc((void **)&device_arr, bytes), "Erro ao alocar device_arr");
    checkCuda(cudaMalloc((void **)&device_somas_prefixos, bytes), "Erro ao alocar device_somas_prefixos");
    checkCuda(cudaMalloc((void **)&device_somas_blocos, bytes_blocos), "Erro ao alocar device_somas_blocos");

    checkCuda(cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice),
              "Erro ao copiar entrada para GPU");

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    checkCuda(cudaEventCreate(&gpu_start), "Erro ao criar evento inicial");
    checkCuda(cudaEventCreate(&gpu_stop), "Erro ao criar evento final");

    dim3 dimGrid(num_blocos, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    checkCuda(cudaEventRecord(gpu_start), "Erro ao iniciar medicao CUDA");
    somaPrefixosBlocos<<<dimGrid, dimBlock>>>(device_arr, device_somas_prefixos, device_somas_blocos, N);
    checkCuda(cudaGetLastError(), "Erro no kernel somaPrefixosBlocos");

    somaPrefixosBlocosPequeno<<<1, BLOCK_SIZE>>>(device_somas_blocos, num_blocos);
    checkCuda(cudaGetLastError(), "Erro no kernel somaPrefixosBlocosPequeno");

    adicionaOffsets<<<dimGrid, dimBlock>>>(device_somas_prefixos, device_somas_blocos, N);
    checkCuda(cudaGetLastError(), "Erro no kernel adicionaOffsets");

    checkCuda(cudaEventRecord(gpu_stop), "Erro ao finalizar medicao CUDA");
    checkCuda(cudaEventSynchronize(gpu_stop), "Erro ao sincronizar GPU");

    float gpu_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop), "Erro ao calcular tempo CUDA");

    checkCuda(cudaMemcpy(host_somas_gpu.data(), device_somas_prefixos, bytes, cudaMemcpyDeviceToHost),
              "Erro ao copiar resultado para CPU");

    int erros = 0;
    for (int i = 0; i < N; i++)
    {
        if (host_somas_cpu[i] != host_somas_gpu[i])
        {
            if (erros < 10)
            {
                std::cerr << "Erro no indice " << i << ": CPU = " << host_somas_cpu[i]
                          << ", GPU = " << host_somas_gpu[i] << "\n";
            }
            erros++;
        }
    }

    std::cout << "N: " << N << "\n";
    std::cout << "tempo_cpu_ms: " << cpu_ms.count() << "\n";
    std::cout << "tempo_cuda_kernels_ms: " << gpu_ms << "\n";
    std::cout << "ultimo_prefixo_cpu: " << host_somas_cpu[N - 1] << "\n";
    std::cout << "ultimo_prefixo_gpu: " << host_somas_gpu[N - 1] << "\n";

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaFree(device_arr);
    cudaFree(device_somas_prefixos);
    cudaFree(device_somas_blocos);

    if (erros > 0)
    {
        std::cerr << "Verificacao falhou: " << erros << " diferencas encontradas.\n";
        return EXIT_FAILURE;
    }

    std::cout << "SOMA DE PREFIXOS OCORREU COM SUCESSO.\n";

    return EXIT_SUCCESS;
}
