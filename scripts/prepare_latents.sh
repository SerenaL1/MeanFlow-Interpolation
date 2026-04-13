#!/bin/bash
#SBATCH --job-name=prep_latents
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_latents_%j.log

source ~/.bashrc
module load cudnn/9.10.2.21_cuda12-fasrc01
conda activate meanflow

cd /n/home06/serenaliu/meanflow

CUDA_VISIBLE_DEVICES=0 python prepare_dataset.py \
    --imagenet_root="/n/netscratch/kempner_undergrads/Lab/serenaliu/imagenet/imagenet_50k" \
    --output_dir="/n/netscratch/kempner_undergrads/Lab/serenaliu/meanflow/meanflow_latents_50k" \
    --batch_size=8 \
    --image_size=256 \
    --compute_latent=True \
    --compute_fid=True \
    --overwrite=True