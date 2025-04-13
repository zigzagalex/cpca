#!/bin/bash

set -e  # Exit immediately if a command fails

echo "🔧 Compiling with AddressSanitizer..."

gcc -fsanitize=address -g \
tests/test_householder.c src/householder.c \
-I./src \
-I/usr/local/opt/openblas/include \
-L/usr/local/opt/openblas/lib \
-lopenblas \
-o run_tests_householder

gcc -fsanitize=address -g \
tests/test_golub_reinsch.c \
src/golub-reinsch.c \
src/householder.c \
-I./src \
-I/usr/local/opt/openblas/include \
-L/usr/local/opt/openblas/lib \
-lopenblas \
-o run_tests_golub_reinsch


echo "🚀 Running householder tests with AddressSanitizer..."
./run_tests_householder
echo "🚀 Running golub-reinsch tests with AddressSanitizer..."
./run_tests_golub_reinsch

