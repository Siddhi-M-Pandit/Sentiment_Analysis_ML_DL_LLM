#!/bin/bash

#export CUDA_VISIBLE_DEVICES="" 

# ----------- Configuration ----------- #
# Number of agents to run
NUM_AGENTS=1
# My Sweep ID
SWEEP_ID="smpandit-/sentiment-analysis/03joa3ol"          #CNN: 'op2fzrj4', # BiLSTM: '03joa3ol'
# Total no. of runs for each agent
NO_RUNS=1

# ------------------------------------- #

# Launch W&B agents in background
for i in $(seq 1 $NUM_AGENTS); do
  echo "Starting agent $i... (will run $NO_RUNS trials)…"
  nohup python -m wandb agent --count $NO_RUNS $SWEEP_ID > wandb_agent_logs/agent_$i.log 2>&1 &
done

wait  

