#!/bin/bash

# Script to run infra-bench-llm inside a running container
# Usage: ./run_container_benchmark.sh <container_name> [benchmark_args]
# Example: ./run_container_benchmark.sh pgai-ollama --models gemma3:1b --iterations 3

set -e

# Check if container name is provided
if [ -z "$1" ]; then
    echo "Error: Container name is required"
    echo "Usage: ./run_container_benchmark.sh <container_name> [benchmark_args]"
    exit 1
fi

CONTAINER_NAME="$1"
shift  # Remove container name from args, leaving only benchmark args

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running"
    echo "Please start the container first"
    exit 1
fi

echo "Running benchmark in container '${CONTAINER_NAME}'..."

# Create a temporary directory in the container
CONTAINER_DIR="/tmp/infra-bench-llm"
echo "Creating directory ${CONTAINER_DIR} in container..."
docker exec "${CONTAINER_NAME}" bash -c "mkdir -p ${CONTAINER_DIR}"

# Copy application files to the container
echo "Copying application files to container..."
docker cp ./src "${CONTAINER_NAME}:${CONTAINER_DIR}/"
docker cp ./requirements.txt "${CONTAINER_NAME}:${CONTAINER_DIR}/"
docker cp ./run_benchmark.py "${CONTAINER_NAME}:${CONTAINER_DIR}/"

# Create visuals directory in container
docker exec "${CONTAINER_NAME}" bash -c "mkdir -p ${CONTAINER_DIR}/visuals"

# Install python3-venv and create virtual environment
echo "Installing python3-venv and creating virtual environment..."
# Try with apt-get first, if it fails (no sudo or apt), try to continue without it
docker exec "${CONTAINER_NAME}" bash -c "cd ${CONTAINER_DIR} && (apt-get update && apt-get install -y python3-pip python3-venv curl || echo 'Warning: Could not install packages. Continuing with existing Python installation...')"

# Create virtual environment (continue even if previous step failed)
echo "Creating virtual environment..."
docker exec "${CONTAINER_NAME}" bash -c "cd ${CONTAINER_DIR} && (python3 -m venv venv || echo 'Warning: Could not create venv. Will use system Python instead.')"

# Install dependencies (try with venv if available, otherwise use system Python)
echo "Installing dependencies..."
docker exec "${CONTAINER_NAME}" bash -c "cd ${CONTAINER_DIR} && if [ -d venv ]; then . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; else pip3 install --user -r requirements.txt; fi"

# Run the benchmark (try with venv if available, otherwise use system Python)
echo "Running benchmark in container..."
docker exec "${CONTAINER_NAME}" bash -c "cd ${CONTAINER_DIR} && if [ -d venv ]; then . venv/bin/activate && python3 run_benchmark.py; else python3 run_benchmark.py; fi"

# Copy results back to host
echo "Copying results back to host..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./container_results/${CONTAINER_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Copy visuals
docker cp "${CONTAINER_NAME}:${CONTAINER_DIR}/visuals/." "${RESULTS_DIR}/"

echo "Benchmark completed successfully!"
echo "Results saved to ${RESULTS_DIR}"

# Optional: Clean up temporary directory in container
# Uncomment the following line if you want to clean up after running
# docker exec "${CONTAINER_NAME}" bash -c "rm -rf ${CONTAINER_DIR}"
