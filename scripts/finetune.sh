#!/bin/bash
#SBATCH --job-name=mf_lora_ft
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_%j.log

source ~/.bashrc
module load cudnn/9.10.2.21_cuda12-fasrc01
conda activate meanflow

cd /n/home06/serenaliu/meanflow

export WORKDIR="/n/netscratch/kempner_undergrads/Lab/serenaliu/meanflow/outputs/lora_finetune_mf_b4"
mkdir -p $WORKDIR

python main.py \
    --workdir=$WORKDIR \
    --config=configs/load_config.py:finetune
