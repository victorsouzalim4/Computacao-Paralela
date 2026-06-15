# Atividade 10 - Perf e Crivo de Eratostenes

Arquivos:

- `crivo_seq.c`: versao sequencial do crivo.
- `crivo_omp2.c`: versao paralela OpenMP fixada em 2 threads.
- `run_perf.sh`: compila e executa as duas versoes com `perf stat`.
- `relatorio_perf_crivo.pdf`: relatorio em PDF.

Como coletar as metricas em Linux:

```bash
cd ativ10
chmod +x run_perf.sh
./run_perf.sh 100000000
```

Observacao: o ambiente atual desta maquina e Windows, sem `gcc`, sem `perf` e sem distribuicao WSL instalada. Por isso, o PDF registra os comandos reprodutiveis e a analise dos gargalos, mas nao inventa valores de medicao.
