#!/bin/bash
#SBATCH --account=def-branzana     
#SBATCH --time=0-11:00      
#SBATCH --gres=gpu:4       #4 = no of gpus, any type 
#SBATCH --tasks-per-node=4 #= number of gpus   
#SBATCH --mem=32G        
#SBATCH --job-name=ct-SSeg
#SBATCH --output=%N-%j.out    #Output from the job is redirected here

module purge
module load python/3.8 scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt 

#nproc_per_node=# of GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py \
        --dataset camvid \
        --cv 2 \
        --arch network.deepv3.DeepWV3Plus \
        --snapshot ./pretrained_models/camvid_best.pth \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 720 \
        --max_cu_epoch 150 \
        --lr 0.002 \
        --lr_schedule scl-poly \
        --poly_exp 1.0 \
        --repoly 1.5  \
        --rescale 1.0 \
        --syncbn \
        --sgd \
        --crop_size 640 \
        --scale_min 0.8 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --gblur \
        --max_epoch 120 \
        --jointwtborder \
        --strict_bdr_cls 2,6,7,9,10 \
        --rlx_off_epoch 100 \
        --wt_bound 1.0 \
        --bs_mult 2 \
        --distributed \
        --exp camvid_ft \
        --ckpt ./logs/ \
        --tb_path ./logs/
