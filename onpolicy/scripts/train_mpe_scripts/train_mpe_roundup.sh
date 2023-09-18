#!/bin/sh
env="MPE"
scenario="simple_formation"
num_landmarks=0
num_good_agents=1
num_adversaries=3 # policy agents
num_agents=3 # also only consider policy agents
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
clip_param=0.2
max_grad_norm=10.0
gamma=0.985
hidden_size=32

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES='2' python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 4 --n_rollout_threads 32 --num_mini_batch 16 --episode_length 200 --num_env_steps 5120000 \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --clip_param ${clip_param} --max_grad_norm ${max_grad_norm} \
    --gamma ${gamma} --hidden_size ${hidden_size} \
    --ppo_epoch 10 --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --wandb_name "xxx" --user_name "finleygou"
done
# --model_dir "/home/sdc/dachuang_space/Project/MAPPO/onpolicy/scripts/results/MPE/simple_round_up/rmappo/check/wandb/run-20230425_004438-7yq8hviv/files/" \
# base lr: 2e-4, advanced:1e-4
# --model_dir "/home/sdc/dachuang_space/Project/MAPPO/onpolicy/scripts/results/MPE/simple_round_up/rmappo/check/wandb/run-20230514_195321-3rclv6kq/files/" \