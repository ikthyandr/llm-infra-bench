# infra-bench-llm

A Python library and CLI tool to benchmark LLM inference on CPU-only infrastructure using Ollama in a Dockerized Ubuntu 22.04 environment.

> **New in container-bench branch:** Run benchmarks in any existing container with the `run_container_benchmark.sh` script!

## Features

- Runs LLM inference benchmarks in a reproducible Docker container
- Installs Ollama and pulls configurable models (default: `gemma3:1b`)
- Measures tokens per second (TPS) and time to first token (TTFT)
- Configurable number of prompt iterations (default: 2)
- Supports benchmarking multiple models in a single run
- Generates visualizations (bar charts) for TPS and TTFT in the `visuals/` directory
- Modular, extensible Python codebase
- Devcontainer and Dockerfile for easy development

## Quickstart

### Running in an Existing Container

If you have an existing container (e.g., `pgai-ollama`) where you want to run the benchmark:

1. Make sure the container is running
2. Run the benchmark script:
   ```bash
   ./run_container_benchmark.sh pgai-ollama
   ```
   To change the defaul models, you can add models on 'config.py'.
   
3. The script will:
   - Copy the application files to the container
   - Try to install python3-venv and create a virtual environment (with fallback to system Python)
   - Install dependencies (in the virtual environment if available, otherwise using --user)
   - Run the benchmark
   - Copy the results back to the host

4. Results will be printed in the terminal and graphs will be saved in the `container_results/<container_name>_<timestamp>/` directory.

### Running in Devcontainer (VSCode)

1. Open the project in VSCode.
2. When prompted, "Reopen in Container" (requires Docker and the VSCode Dev Containers extension).
3. Once the container is ready, open a terminal and run:
   ```bash
   python3 run_benchmark.py
   ```
   This will use the default models and prompt.

4. Results will be printed in the terminal and graphs will be saved in the `visuals/` directory.

### Running as a Standalone Library

1. Ensure you have Python 3.8+ installed.
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Make sure Ollama is installed and running, or let the script install it for you.
4. Run the benchmark:
   ```bash
   python3 run_benchmark.py
   ```
   You can also specify models, iterations, and prompt:
   ```bash
   python3 run_benchmark.py --models gemma3:1b llama2:7b --iterations 3 --prompt "Your prompt here"
   ```

### CLI Usage

```bash
python3 run_benchmark.py --models gemma3:1b llama2:7b --iterations 3 --prompt "Your prompt here"
```

#### Configuration Options

- `--models`: List of Ollama model names to benchmark (default: as defined in `src/infra_bench_llm/config.py`)
- `--iterations`: Number of prompt iterations (default: 2)
- `--prompt`: Prompt to use for benchmarking

## Project Structure

- `src/infra_bench_llm/`: Core benchmarking library and configuration
- `run_benchmark.py`: CLI runner
- `visuals/`: Output directory for generated graphs
- `Dockerfile`, `.devcontainer/`: Development environment

## Output

After running, a report is printed showing TPS, TTFT, and other metrics for each model and each iteration. Bar charts for TPS and TTFT are saved in the `visuals/` directory as `tps.png` and `ttft.png`.

## License

MIT
