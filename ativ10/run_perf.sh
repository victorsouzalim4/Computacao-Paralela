#!/usr/bin/env bash
set -euo pipefail

N="${1:-100000000}"

gcc -O3 -march=native crivo_seq.c -lm -o crivo_seq
gcc -O3 -march=native -fopenmp crivo_omp2.c -lm -o crivo_omp2

EVENTS="task-clock,cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend,LLC-loads,LLC-load-misses"

echo "== Sequencial =="
perf stat -r 3 -e "$EVENTS" ./crivo_seq "$N"

echo "== Paralelo OpenMP: 2 threads =="
OMP_NUM_THREADS=2 perf stat -r 3 -e "$EVENTS" ./crivo_omp2 "$N"
