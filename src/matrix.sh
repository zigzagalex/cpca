#!/bin/bash

echo "Running compiler"

gcc matrices.c \
-I./src \
-I/usr/local/opt/openblas/include \
-L/usr/local/opt/openblas/lib \
-lopenblas \
-o run_matrices

echo "Running binary"
./run_matrices
