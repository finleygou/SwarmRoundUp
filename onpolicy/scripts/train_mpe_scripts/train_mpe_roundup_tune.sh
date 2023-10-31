#!/bin/sh
env="MPE"
scenario="simple_round_up2_tune"
num_landmarks=4
num_good_agents=1
num_adversaries=5 # policy agents
num_agents=5 # also only consider policy agents
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
clip_param=0.15
max_grad_norm=10.0
gamma=0.985
hidden_size=32
decay_ratio=0.0
use_linear_lr_decay=False
save_data=False  # needs to be modified in config.py

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES='2,3' python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 8 --n_rollout_threads 32 --num_mini_batch 32 --episode_length 200 --num_env_steps 5120000 \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --clip_param ${clip_param} --max_grad_norm ${max_grad_norm} \
    --gamma ${gamma} --hidden_size ${hidden_size} --save_data ${save_data} --use_linear_lr_decay ${use_linear_lr_decay} --decay_ratio ${decay_ratio} \
    --model_dir "/home/sdc/goufandi_space/Projects/SwarmRoundUp_V2/onpolicy/scripts/results/MPE/simple_round_up2/rmappo/check/wandb/run-20231029_112243-1x3u5pkr/files/"\
    --ppo_epoch 15 --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --wandb_name "xxx" --user_name "finleygou"
done
# tune 37 base: run-20231026_222247-3kflqiwv
# tune 38 base: run-20231029_112243-1x3u5pkr