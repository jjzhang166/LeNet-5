@echo off
clang -Wno-deprecated-declarations -O3 -mavx -fopenmp  batch.c predict.c main.c -o LeNet_5.exe
if exist LeNet_5.exe editbin /Stack:8000000 LeNet_5.exe