#!/bin/bash

# ----------- Configuration ----------- #
NUM_AGENTS=1                                 # how many agents to spin up
NO_RUNS=1                                    # how many trials each agent should run
ENTITY="smpandit-"                           # your W&B entity (note the trailing dash)
PROJECT="sentiment-analysis"                 # your W&B project name
SWEEP_ID="rnz1o0g9"                          # your LLM sweep ID
LOG_DIR="wandb_agent_logs"                   # where to stash agent output

# ------------------------------------- #

# Launch W&B agents in background
for i in $(seq 1 $NUM_AGENTS); do
  echo "Starting LLM agent $i… (will run $NO_RUNS trials)"
  nohup python -m wandb agent \
    --entity $ENTITY \
    --project $PROJECT \
    --count $NO_RUNS \
    $SWEEP_ID \
    > $LOG_DIR/agent_${i}.log 2>&1 &
done
wait