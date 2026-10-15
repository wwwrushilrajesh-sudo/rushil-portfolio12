#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvcc -O2 -std=c++17 --expt-relaxed-constexpr --fmad=false -shared -Xcompiler -fPIC backend.cu -o libcylinder_solver.so -lcufft -lcusolver -lcublas
