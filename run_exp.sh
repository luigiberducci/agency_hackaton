#!/bin/bash

logdir=logs/

# create list of exp arguments
args=(
  # exp-name script env-id num-envs tot-steps reward
  #
  "one-door train.py one-door-2-agents-v0 4 1000000 sparse"
  "one-door train.py one-door-2-agents-v0 4 1000000 altruistic"
  "one-door train.py one-door-2-agents-v0 4 1000000 neg_distance"
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
nenvs=$(echo $exp_args | cut -d' ' -f4)   # num-envs for vectorization
nsteps=$(echo $exp_args | cut -d' ' -f5)  # total training steps
reward=$(echo $exp_args | cut -d' ' -f6)  # reward type as defined in envs.reward_wrappers.reward_factory


cmd="
python ${script} --log-dir ${logdir}/${exp_name} --env-id ${env_id} --num-envs ${nenvs} \
                 --total-timesteps ${nsteps} --reward-type ${reward}
"

echo $cmd
eval $cmd
