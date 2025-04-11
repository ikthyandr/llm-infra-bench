import argparse
import shutil
import subprocess
import sys

from src.infra_bench_llm.benchmark import BenchmarkConfig, BenchmarkRunner, print_report
from src.infra_bench_llm.config import (
    DEFAULT_MODELS,
    DEFAULT_ITERATIONS,
    DEFAULT_PROMPT,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference benchmark in Docker/Ollama.")
    parser.add_argument("--models", type=str, nargs="*", default=None, help=f"List of Ollama model names (e.g. gemma3:1b llama2:7b). If not provided, uses --model or default ({DEFAULT_MODELS}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODELS[0], help=f"(Deprecated) Ollama model name (default: {DEFAULT_MODELS[0]}). Use --models instead for multiple models.")
    parser.add_argument("--quantizations", type=str, nargs="*", default=None, help="List of quantizations (e.g. q4 q8). If not provided, runs with no quantization.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help=f"Number of prompt iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help=f"Prompt to use (default: '{DEFAULT_PROMPT}')")
    return parser.parse_args()

def main():
    # Check if ollama is installed
    if shutil.which("ollama") is None:
        print("Ollama not found. Installing Ollama...")
        try:
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["sh"],
                input=subprocess.run(
                    ["curl", "-fsSL", "https://ollama.com/install.sh"],
                    check=True,
                    stdout=subprocess.PIPE
                ).stdout,
                check=True
            )
        except Exception as e:
            print(f"Failed to install Ollama: {e}")
            sys.exit(1)
        if shutil.which("ollama") is None:
            print("Ollama installation failed or not found in PATH.")
            sys.exit(1)
        print("Ollama installed successfully.")

    args = parse_args()
    # Determine models list: prefer --models, fallback to --model, else use DEFAULT_MODELS
    if args.models is not None and len(args.models) > 0:
        models = args.models
    elif "model" in args and args.model != DEFAULT_MODELS[0]:
        models = [args.model]
    else:
        models = DEFAULT_MODELS

    config = BenchmarkConfig(
        models=models,
        quantizations=args.quantizations,
        iterations=args.iterations,
        prompt=args.prompt
    )
    runner = BenchmarkRunner(config)
    results = runner.run()
    print_report(results)

if __name__ == "__main__":
    main()
