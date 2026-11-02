#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -o job64_%x_%j.out
# usage: sbatch -J <name> job64_CASTNER.sh <gpu_vanilla|gpu_restrict|cpu_vanilla>

cd /home/users/e75014/DDalphaAMG
source env_CASTNER.sh
V=/home/users/e75014/vanilla_DDalphaAMG

case "$1" in
  gpu_vanilla)
    srun -n 2 $MPIRUN_OPTS $V/dd_alpha_amg_gpu_vanilla 3lvl_sample64_CASTNER.ini > result64_gpu_vanilla.txt 2>&1 ;;
  gpu_restrict)
    srun -n 2 $MPIRUN_OPTS ./dd_alpha_amg_gpu_restrict 3lvl_sample64_CASTNER.ini > result64_gpu_restrict.txt 2>&1 ;;
  cpu_vanilla)
    MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=72 srun -n 2 -c 72 --cpu-bind=cores \
        $V/dd_alpha_amg_cpu_vanilla 3lvl_sample64_CASTNER_cpu.ini > result64_cpu_vanilla.txt 2>&1 ;;
  gpu_restrict_nsys)
    srun -n 2 $MPIRUN_OPTS nsys profile --trace=cuda,nvtx --force-overwrite=true -o prof64_restrict_rank%q{SLURM_PROCID} ./dd_alpha_amg_gpu_restrict 3lvl_sample64_CASTNER.ini > result64_gpu_restrict_nsys.txt 2>&1 ;;
  gpu_restrict_nsys)
    srun -n 2 $MPIRUN_OPTS nsys profile --trace=cuda,nvtx --force-overwrite=true -o prof64_restrict_rank%q{SLURM_PROCID} ./dd_alpha_amg_gpu_restrict 3lvl_sample64_CASTNER.ini > result64_gpu_restrict_nsys.txt 2>&1 ;;
  gpu_restrict_test)
    srun -n 2 $MPIRUN_OPTS ./dd_alpha_amg_gpu_restrict_test 3lvl_sample64_CASTNER.ini > result64_gpu_restrict_test.txt 2>&1 ;;
  gpu_vanilla_omp72)
    OMP_NUM_THREADS=72 srun -n 2 -c 72 --cpu-bind=cores --gpus-per-task=1 $V/dd_alpha_amg_gpu_vanilla 3lvl_sample64_CASTNER_omp72.ini > result64_gpu_vanilla_omp72.txt 2>&1 ;;
  gpu_restrict_omp72)
    OMP_NUM_THREADS=72 srun -n 2 -c 72 --cpu-bind=cores --gpus-per-task=1 ./dd_alpha_amg_gpu_restrict 3lvl_sample64_CASTNER_omp72.ini > result64_gpu_restrict_omp72.txt 2>&1 ;;
  *)
    echo "unknown run: $1"; exit 1 ;;
esac
echo "=== $1 done ==="
