#!/bin/bash

# ----------- Configuration ----------- #
# Number of agents to run
NUM_AGENTS=3
# My Sweep ID
SWEEP_ID="smpandit-/sentiment-analysis/wuh1gf90"
# Total no. of runs
TOTAL_RUNS=7

# ------------------------------------- #

# Launch W&B agents in background
for i in $(seq 1 $NUM_AGENTS); do
  echo "Starting agent $i..."
  WANDB_DIR="./wandb_agent_logs/agent_$i" nohup wandb agent $SWEEP_ID > agent_$i.log 2>&1 &
done

wait  
