#!/bin/sh
env="MPE"
scenario="simple_round_up"
num_landmarks=0
num_good_agents=1
num_adversaries=5 # policy agents
num_agents=5 # also only consider policy agents
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
clip_param=0.12
max_grad_norm=4.0
gamma=0.985
hidden_size=128

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES='2,3' python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 4 --n_rollout_threads 32 --num_mini_batch 16 --episode_length 200 --num_env_steps 6000000 \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --clip_param ${clip_param} --max_grad_norm ${max_grad_norm} \
    --gamma ${gamma} --hidden_size ${hidden_size} \
    --ppo_epoch 10 --gain 0.01 --lr 2e-5 --critic_lr 2e-5 --wandb_name "xxx" --user_name "finleygou"
done