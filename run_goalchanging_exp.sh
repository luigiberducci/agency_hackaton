#!/bin/bash

logdir=logs/changing-goals-sparse

# create list of exp arguments
args=(
  # exp-id script env-id hide-goals nenvs nsteps reward goal_changing_interval stack_frames
  "changing-goal-FS4-0.01 train.py two-door-2-agents-goal-change-v0 True 4 1000000 sparse 0.01 4"
  "changing-goal-FS4-0.0 train.py two-door-2-agents-goal-change-v0 True 4 1000000 sparse 0.0 4"
  "changing-goal-FS0-0.01 train.py two-door-2-agents-goal-change-v0 True 4 1000000 sparse 0.01 0"
  "changing-goal-FS4-0.01 train.py two-door-2-agents-goal-change-v0 False 4 1000000 sparse 0.01 4"
  "changing-goal-FS4-0.05 train.py two-door-2-agents-goal-change-v0 True 4 1000000 sparse 0.05 4"
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
goal_changing_interval=$(echo $exp_args | cut -d' ' -f8)  
stack_frames=$(echo $exp_args | cut -d' ' -f9)  


cmd="
python ${script} --log-dir ${logdir}/${exp_name} --env-id ${env_id} --num-envs ${nenvs} \
                 --total-timesteps ${nsteps} --reward ${reward} --hide-goals=${hide_goals} --goal-changes=${goal_changing_interval} --stack-frames=${stack_frames}
"

echo $cmd
eval $cmd
