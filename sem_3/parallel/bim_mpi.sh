#!/bin/bash
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=36

#SBATCH --time=2:00:00

#SBATCH --output=results%j.out



mpirun ./a.out 2000 1000 1000 
mpirun ./a.out 4000 2000 2000 



