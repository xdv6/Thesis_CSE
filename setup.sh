#!/bin/bash

cd $WORKDIR_PATH/Thesis_CSE/reward_machines/reward_machines
source $WORKDIR_PATH/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python run_robosuite.py --env=MyBlockStackingEnvRM1 --num_timesteps=100000 --alg=dhrm