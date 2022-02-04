#!/bin/sh

#SBATCH -J wm_initial
#SBATCH -o out.wm_initial
#SBATCH -p gpu15
#SBATCH -t 4-00:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1

module purge
module load cuda/cuda-8.0.cudnn-7.0.5

echo $SLURM_SUBMIT_HOSE
echo $SLURM_JOB_NODELIST
echo $SLURM_SUBMIT_DIR

echo ###START###

/home/hwihun/.conda/envs/watermark/bin/python /home/hwihun/Deep-Model-Watermarking/Initial_stage/main.py --datasets '../data_wm/LR_1e-05_epoch353/' --gpu_ind '0,1,2,3' --ngpu 4 --batchSize 8
# End of File.
