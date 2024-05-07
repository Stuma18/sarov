#!/bin/bash
# Пример запуска OpenMP приложения с использованием 8 ядер.
# 
# Ограничение времени выполнения задачи, например, 1 час.
#SBATCH --time=1:00:00
#
# Использовать 6 узел.
##SBATCH --nodes=6
#
# Максимальное количество ядер, которое будет использоватся в программе.(может быть установленно от 1 до 36) 
#SBATCH --ntasks-per-node=36
#
# 
# Загрузка модулей.
# module load gcc/gcc-7.4
#
# Запуск программы OpenMP
OMP_NUM_THREADS=1 ./a.out 4000 2000 2000
OMP_NUM_THREADS=2 ./a.out 4000 2000 2000
OMP_NUM_THREADS=4 ./a.out 4000 2000 2000
OMP_NUM_THREADS=8 ./a.out 4000 2000 2000
OMP_NUM_THREADS=16 ./a.out 4000 2000 2000
OMP_NUM_THREADS=32 ./a.out 4000 2000 2000
OMP_NUM_THREADS=36 ./a.out 4000 2000 2000


