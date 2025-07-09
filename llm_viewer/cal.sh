#!/bin/bash

# --- Configuration ---
# Define the lists of batch sizes and sequence lengths to iterate over
BATCH_SIZES=(1 2 4 8 16 32)
SEQ_LENS=(128 256 512 1024 2048 4096)

# --- Script Logic ---
echo "Starting analysis runs..."

PYTHON_EXE="python3"
# HARDWARE="nvidia_H100"
# HARDWARE="nvidia_A100_80G"
HARDWARE="nvidia_H100_PCIe"
# HARDWARE="nvidia_A100"

# Define the base command components
SCRIPT_PATH="analyze_cli.py"
MODEL_PATH="../Llama"

# Outer loop for batch sizes
for bs in "${BATCH_SIZES[@]}"; do
  # Inner loop for sequence lengths
  for sl in "${SEQ_LENS[@]}"; do
    echo "--------------------------------------------------"
    echo "Running analysis with Batch Size = $bs and Sequence Length = $sl"
    echo "--------------------------------------------------"

    # Construct the full command
    COMMAND="$PYTHON_EXE $SCRIPT_PATH $MODEL_PATH $HARDWARE --batchsize $bs --seqlen $sl"

    # Print the command being executed (optional)
    echo "Executing: $COMMAND"

    # Execute the command
    # 执行命令
    $COMMAND

    # Add a small delay if needed, e.g., sleep 1
    echo "Finished run for BS=$bs, SL=$sl."
    echo "" # Add a blank line for better readability 
  done
done

# Define the base command components
SCRIPT_PATH="analyze_cli_gla.py"
MODEL_PATH="../gta"

# Outer loop for batch sizes
for bs in "${BATCH_SIZES[@]}"; do
  # Inner loop for sequence lengths
  for sl in "${SEQ_LENS[@]}"; do
    echo "--------------------------------------------------"
    echo "Running analysis with Batch Size = $bs and Sequence Length = $sl"
    echo "--------------------------------------------------"

    # Construct the full command
    COMMAND="$PYTHON_EXE $SCRIPT_PATH $MODEL_PATH $HARDWARE --batchsize $bs --seqlen $sl"

    # Print the command being executed (optional)
    echo "Executing: $COMMAND"

    # Execute the command
    $COMMAND

    # Add a small delay if needed, e.g., sleep 1
    echo "Finished run for BS=$bs, SL=$sl."
    echo "" # Add a blank line for better readability 
  done
done

echo "--------------------------------------------------"
echo "All analysis runs completed."
echo "--------------------------------------------------"

# --- End of Script ---
