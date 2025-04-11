import time
import subprocess
import requests
from typing import List, Dict, Optional, Union
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
import os

from .config import (
    DEFAULT_MODELS,
    DEFAULT_ITERATIONS,
    DEFAULT_PROMPT,
)

class BenchmarkConfig:
    def __init__(
        self,
        models: Optional[List[str]] = None,
        quantizations: Optional[List[str]] = None,
        iterations: int = DEFAULT_ITERATIONS,
        prompt: str = DEFAULT_PROMPT,
    ):
        # Accepts either a single model as string or a list of models
        if models is None:
            self.models = DEFAULT_MODELS
        elif isinstance(models, str):
            self.models = [models]
        else:
            self.models = models
        # If quantizations is None, run with [None] (no quantization)
        self.quantizations = quantizations if quantizations is not None else [None]
        self.iterations = iterations
        self.prompt = prompt

class BenchmarkResult:
    def __init__(
        self,
        model: str,
        quantization: str,
        tps: float,
        ttft: float,
        total_tokens: int,
        times: List[float],
        tokens_per_iteration: List[int],
    ):
        self.model = model
        self.quantization = quantization
        self.tps = tps
        self.ttft = ttft
        self.total_tokens = total_tokens
        self.times = times
        self.tokens_per_iteration = tokens_per_iteration

    def as_dict(self):
        return {
            "model": self.model,
            "quantization": self.quantization,
            "tokens_per_second": self.tps,
            "time_to_first_token": self.ttft,
            "total_tokens": self.total_tokens,
            "iteration_times": self.times,
            "tokens_per_iteration": self.tokens_per_iteration,
        }

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def _get_ollama_model_name(self, model: str, quantization: Optional[str]) -> str:
        if quantization:
            return f"{model}:{quantization}"
        return model

    def pull_model(self, model: str, quantization: Optional[str]):
        model_name = self._get_ollama_model_name(model, quantization)
        print(f"Pulling model '{model_name}' ...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull model: {result.stderr}")

    def start_ollama_server(self):
        # Start Ollama server in background if not already running
        print("Starting Ollama server ...")
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("localhost", 11434))
            print("Ollama server already running.")
            s.close()
            return
        except Exception:
            pass
        subprocess.Popen(["ollama", "serve"])
        time.sleep(3)  # Wait for server to start

    def run_prompt(self, model: str, quantization: Optional[str], prompt: str) -> Dict:
        """
        Returns:
            {
                "text": full generated text,
                "elapsed": total time to receive all tokens,
                "ttft": time to first token
            }
        """
        import json

        model_name = self._get_ollama_model_name(model, quantization)
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        start_time = time.time()
        first_token_time = None
        generated_text = ""
        try:
            with requests.post(url, headers=headers, json=data, stream=True, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    # Ollama streams JSON lines
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if "response" in obj:
                        if first_token_time is None:
                            first_token_time = time.time()
                        generated_text += obj["response"]
            elapsed = time.time() - start_time
            ttft = (first_token_time - start_time) if first_token_time is not None else elapsed
        except Exception as e:
            raise RuntimeError(f"Prompt failed: {e}")
        return {
            "text": generated_text,
            "elapsed": elapsed,
            "ttft": ttft
        }

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def run(self) -> List['BenchmarkResult']:
        self.start_ollama_server()
        results = []
        for model in self.config.models:
            for quant in self.config.quantizations:
                self.pull_model(model, quant)
                times = []
                tokens_per_iteration = []
                ttft_per_iteration = []
                total_tokens = 0
                for i in range(self.config.iterations):
                    print(f"Running prompt iteration {i+1} for model '{model}' quantization '{quant}'...")
                    result = self.run_prompt(model, quant, self.config.prompt)
                    tokens = self.count_tokens(result["text"])
                    tokens_per_iteration.append(tokens)
                    total_tokens += tokens
                    times.append(result["elapsed"])
                    ttft_per_iteration.append(result.get("ttft", result["elapsed"]))
                tps = total_tokens / sum(times) if sum(times) > 0 else 0
                # For backward compatibility, keep ttft as first iteration's TTFT
                results.append(BenchmarkResult(
                    model=model,
                    quantization=quant or "default",
                    tps=tps,
                    ttft=ttft_per_iteration[0] if ttft_per_iteration else 0.0,
                    total_tokens=total_tokens,
                    times=times,
                    tokens_per_iteration=tokens_per_iteration
                ))
                # Attach per-iteration TTFT for reporting
                results[-1].ttft_per_iteration = ttft_per_iteration
        return results

def plot_results(results: List[BenchmarkResult], visuals_dir="visuals"):

    # Ensure visuals directory exists
    os.makedirs(visuals_dir, exist_ok=True)

    # Collect unique models and assign colors
    models = list({res.model for res in results})
    color_map = plt.get_cmap("tab10")
    model_colors = {model: color_map(i % 10) for i, model in enumerate(models)}

    # Prepare data: group results by model
    model_to_results = {}
    for res in results:
        model_to_results.setdefault(res.model, []).append(res)

    # Helper to get max number of iterations
    max_iters = max(len(res.times) for res in results)

    # Bar plot for TPS
    plt.figure(figsize=(8, 5))
    bar_width = 0.8 / max(len(models), 1)
    x = np.arange(1, max_iters + 1)
    for idx, (model, res_list) in enumerate(model_to_results.items()):
        for res in res_list:
            tps_list = [round((res.tokens_per_iteration[i] / res.times[i]), 2) if res.times[i] > 0 else 0.0 for i in range(len(res.times))]
            # Pad to max_iters for alignment
            tps_list += [0.0] * (max_iters - len(tps_list))
            label = f"{model}" if len(model_to_results) == 1 else f"{model} ({res.quantization})"
            plt.bar(x + idx * bar_width - (bar_width * (len(models)-1) / 2), tps_list, width=bar_width, label=label, color=model_colors[model])
    plt.xlabel("Iteration")
    plt.ylabel("Tokens Per Second (TPS)")
    plt.title("Tokens Per Second (TPS) per Iteration")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "tps.png"))
    plt.close()

    # Bar plot for TTFT
    plt.figure(figsize=(8, 5))
    for idx, (model, res_list) in enumerate(model_to_results.items()):
        for res in res_list:
            ttft_raw = getattr(res, "ttft_per_iteration", [res.ttft] + [0.0] * (len(res.times)-1))
            ttft_list = [round(ttft, 2) if isinstance(ttft, float) else 0.0 for ttft in ttft_raw]
            ttft_list += [0.0] * (max_iters - len(ttft_list))
            label = f"{model}" if len(model_to_results) == 1 else f"{model} ({res.quantization})"
            plt.bar(x + idx * bar_width - (bar_width * (len(models)-1) / 2), ttft_list, width=bar_width, label=label, color=model_colors[model])
    plt.xlabel("Iteration")
    plt.ylabel("Time To First Token (TTFT) [s]")
    plt.title("Time To First Token (TTFT) per Iteration")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "ttft.png"))
    plt.close()

def print_report(results: List[BenchmarkResult]):
    print("Benchmark Report:")
    for res in results:
        print(f"Model: {res.model}")
        times_list = [round(t, 2) for t in res.times]
        tokens_list = [int(tk) for tk in res.tokens_per_iteration]
        tps_list = [round((res.tokens_per_iteration[i] / res.times[i]), 2) if res.times[i] > 0 else 0.0 for i in range(len(res.times))]
        ttft_raw = getattr(res, "ttft_per_iteration", [res.ttft] + [0.0] * (len(res.times)-1))
        ttft_list = [round(ttft, 2) if isinstance(ttft, float) else 0.0 for ttft in ttft_raw]

        print(f"  TPS: {tps_list}")
        print(f"  TTFT: {ttft_list}")
        print(f"  Time elapsed: {times_list}")
        print(f"  Tokens generated: {tokens_list}")
        print("")
    # Plot and save graphs
    plot_results(results)

if __name__ == "__main__":
    config = BenchmarkConfig()
    runner = BenchmarkRunner(config)
    results = runner.run()
    print_report(results)
