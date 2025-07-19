#!/bin/bash

# Read user input for model path and hardware
read -e -p "Enter model path for Llama-GQA (e.g., ../gqa): " MODEL_PATH_CLI
read -e -p "Enter model path for Llama-GTA (e.g., ../gta): " MODEL_PATH_GTA
read -p "Enter hardware identifier (e.g., nvidia_H100, nvidia_A100): " HARDWARE

BATCH_SIZES=(1 2 4 8 16 32)
SEQ_LENS=(128 256 512 1024 2048 4096)

echo "Starting analysis runs..."

PYTHON_EXE="python3"

SCRIPT_PATH="analyze_cli.py"

if [ -d "$MODEL_PATH_CLI" ]; then
  for bs in "${BATCH_SIZES[@]}"; do
    for sl in "${SEQ_LENS[@]}"; do
      echo "--------------------------------------------------"
      echo "Running analyze_cli.py with Batch Size = $bs and Sequence Length = $sl"
      echo "--------------------------------------------------"

      COMMAND="$PYTHON_EXE $SCRIPT_PATH $MODEL_PATH_CLI $HARDWARE --batchsize $bs --seqlen $sl"
      echo "Executing: $COMMAND"
      $COMMAND

      echo "Finished run for BS=$bs, SL=$sl."
      echo ""
    done
  done
else
  echo "Model path for analyze_cli.py not found: $MODEL_PATH_CLI"
  echo "Skipping analyze_cli.py runs..."
fi

# Second set of runs with analyze_cli_gta.py
SCRIPT_PATH="analyze_cli_gta.py"

if [ -d "$MODEL_PATH_GTA" ]; then
  for bs in "${BATCH_SIZES[@]}"; do
    for sl in "${SEQ_LENS[@]}"; do
      echo "--------------------------------------------------"
      echo "Running analyze_cli_gta.py with Batch Size = $bs and Sequence Length = $sl"
      echo "--------------------------------------------------"

      COMMAND="$PYTHON_EXE $SCRIPT_PATH $MODEL_PATH_GTA $HARDWARE --batchsize $bs --seqlen $sl"
      echo "Executing: $COMMAND"
      $COMMAND

      echo "Finished run for BS=$bs, SL=$sl."
      echo ""
    done
  done
else
  echo "Model path for analyze_cli_gta.py not found: $MODEL_PATH_GTA"
  echo "Skipping analyze_cli_gta.py runs..."
fi

echo "--------------------------------------------------"
echo "All analysis runs completed."
echo "--------------------------------------------------"