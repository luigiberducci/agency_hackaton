#!/bin/bash

logdir=logs/

# create list of exp arguments
args=(
  # exp-name script env-id hide-goals num-envs tot-steps reward distr-correction
  #
  # exp one-door with hidden goals
  "one-door train.py one-door-2-agents-v0 True 4 2000000 sparse False"
  "one-door train.py one-door-2-agents-v0 True 4 2000000 altruistic False"
  "one-door train.py one-door-2-agents-v0 True 4 2000000 neg_distance False"
  #
  # exp one-door with visible goals
  "one-door train.py one-door-2-agents-v0 False 4 2000000 sparse False"
  "one-door train.py one-door-2-agents-v0 False 4 2000000 altruistic False"
  "one-door train.py one-door-2-agents-v0 False 4 2000000 neg_distance False"
  #
  #
  # exp two-doors with hidden goals, skewed goal distribution towards top row
  "two-doors train.py two-doors-2-agents-skewed-v0 True 4 2000000 neg_distance False"
  #
  # exp two-doors with hidden goals, skewed goal distribution towards bottom row
  "two-doors train.py two-doors-2-agents-skewed-v1 True 4 2000000 neg_distance False"
  #
  # exp two-doors with hidden goals, skewed goal distribution towards top row
  "two-doors train.py two-doors-2-agents-skewed-v0 True 4 2000000 neg_distance True"
  #
  # exp two-doors with hidden goals, skewed goal distribution towards bottom row
  "two-doors train.py two-doors-2-agents-skewed-v0 True 4 2000000 neg_distance True"
)

# check the only input is exp-id
if [ $# -ne 1 ]; then
  echo "Usage: $0 exp-id"
  exit 1
fi

# parse args based on exp-id
exp_id=$1

if [ $exp_id -lt 0 ] || [ $exp_id -ge ${#args[@]} ]; then
  echo "Invalid exp-id: $exp_id"
  exit 1
fi

# extract args
exp_args=${args[$exp_id]}

# parse exp args
exp_name=$(echo $exp_args | cut -d' ' -f1)  # exp-name
script=$(echo $exp_args | cut -d' ' -f2)  # script to launch the exp
env_id=$(echo $exp_args | cut -d' ' -f3)  # env-id as registered in envs/__init__.py
hide_goals=$(echo $exp_args | cut -d' ' -f4)  # whether to hide goals from agents
nenvs=$(echo $exp_args | cut -d' ' -f5)   # num-envs for vectorization
nsteps=$(echo $exp_args | cut -d' ' -f6)  # total training steps
reward=$(echo $exp_args | cut -d' ' -f7)  # reward type as defined in envs.reward_wrappers.reward_factory
distr_correction=$(echo $exp_args | cut -d' ' -f8)  # whether to use distribution correction


cmd="
python ${script} --log-dir ${logdir}/${exp_name} --env-id ${env_id} --num-envs ${nenvs} \
                 --total-timesteps ${nsteps} --reward ${reward} --hide-goals=${hide_goals} \
                 --distr-correction=${distr_correction}
"

echo $cmd
eval $cmd
