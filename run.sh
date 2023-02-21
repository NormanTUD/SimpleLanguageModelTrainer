#!/bin/bash

#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=90000 

set -x

LMOD_DIR=/usr/share/lmod/lmod/libexec/

ml () {
    eval $($LMOD_DIR/ml_cmd "$@")
}

ml modenv/hiera GCCcore/11.3.0 Python/3.9.6 CUDA/11.7.0

source /lustre/scratch2/ws/0/s3811141-sprachmodell_test/env/bin/activate

python3 test.py $@ | tee train_log
