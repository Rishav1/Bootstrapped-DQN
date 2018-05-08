#!/bin/bash

# Change to current directory
cd `dirname -- "$0"`

# Find the experiment type to run.
if [ -z "$1" ]; then
  echo "Please enter the experiment type"
  echo "Atari Choices: atari_bootstrap | atari_swarm"
  exit 0
else
  EXP=$1
  shift
fi

# Find the game to run experiment on
if [ -z "$1" ];then
  echo "Please enter game"
  exit 0
else
  GAME=$1
  shift
fi

if [ "$EXP" == "atari_bootstrap" ]; then
  python train_atari.py --gpu=1 --env=$GAME --bootstrap --heads=5 --num-steps=50000000 --batch-size=32 --target-update-freq=10000 --prioritized --save-dir="/scratch/r.chourasia/baselines_log/bootstrap/" "$@"
elif [ "$EXP" == "atari_swarm" ]; then
  python train_atari.py --gpu=1 --env=$GAME --bootstrap --swarm --heads=5 --num-steps=50000000 --batch-size=32 --target-update-freq=1000 --prioritized --save-dir="/scratch/r.chourasia/baselines_log/swarm/" "$@"
else
  echo "Invalid options"
fi