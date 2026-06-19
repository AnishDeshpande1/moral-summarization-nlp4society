"""Ollama backend for moral summarization inference.

This provides a drop-in replacement for ``LlamaModelForSequenceCompletion`` that
talks to a local Ollama server (https://ollama.com) instead of loading the model
in-process with transformers + bitsandbytes.

Why a separate backend:
- On low-VRAM machines (e.g. a 6 GB laptop GPU) a 4-bit 8B model loaded through
  transformers will OOM once the long few-shot prompts fill the KV cache. Ollama
  serves a GGUF quant and transparently offloads layers to CPU, so it runs (if
  slowly) on small GPUs and is trivial to scale up on a bigger machine.
- Importing this module does NOT pull in transformers/peft/bitsandbytes, so it
  works in a lightweight local environment installed with ``pip install -e .
  --no-deps``.

It exposes the same ``get_response(query, system_content, ...)`` signature used by
``prompting.py`` so the inference loop is backend-agnostic.
"""
import time

import requests


class OllamaModelForSequenceCompletion:
    def __init__(self, config):
        self.config = config
        self.verbose = config.get('verbose', True)

        ollama_cfg = config.get('ollama', {})
        # Name of the Ollama model/tag, e.g. "llama3.1:8b-instruct-q4_K_M".
        self.model_name = ollama_cfg.get('model', 'llama3.1:8b-instruct-q4_K_M')
        self.host = ollama_cfg.get('host', 'http://localhost:11434')
        # num_ctx: context window. Few-shot prompts here are long, so default
        # generously. Larger values cost more VRAM/RAM; tune per machine.
        self.num_ctx = ollama_cfg.get('num_ctx', 8192)
        # Request timeout in seconds. CPU-offloaded generation is slow.
        self.request_timeout = ollama_cfg.get('request_timeout', 1200)

        self.chat_url = f"{self.host.rstrip('/')}/api/chat"

        if self.verbose:
            print(f"Using Ollama backend: model={self.model_name} host={self.host}")
        self._check_server()

    def _check_server(self):
        """Fail fast with a helpful message if Ollama isn't reachable."""
        try:
            resp = requests.get(f"{self.host.rstrip('/')}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is the server running? "
                f"Start it with 'ollama serve' (or the desktop app). Original error: {e}"
            )

        tags = [m.get('name') for m in resp.json().get('models', [])]
        if self.model_name not in tags:
            print(
                f"WARNING: model '{self.model_name}' not found in Ollama "
                f"(available: {tags}). Pull it with "
                f"'ollama pull {self.model_name}'. Will attempt anyway; Ollama "
                f"may auto-pull on first request."
            )

    def get_response(
            self,
            query,
            system_content="You are a news summarizer assistant and a moral expert.",
            max_tokens=4096,
            temperature=0.6,
            top_p=0.9,
            ):
        """Generate a response for ``query``.

        Returns ``(response, conversation)`` where ``response`` is a list of
        message dicts (so callers can use ``response[-1]['content']``) and
        ``conversation`` is the full message list including the assistant turn.
        This mirrors ``LlamaModelForSequenceCompletion.get_response``.
        """
        conversation = [{"role": "system", "content": system_content}]
        prompt = conversation + [{"role": "user", "content": query}]

        payload = {
            "model": self.model_name,
            "messages": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "seed": 0,
            },
        }

        if self.verbose:
            print("Prompting the model...")
        start_time = time.time()

        resp = requests.post(self.chat_url, json=payload, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()

        if self.verbose:
            print(f"Execution time: {time.time() - start_time:.1f}s")

        assistant_message = {
            "role": "assistant",
            "content": data["message"]["content"],
        }
        response = [assistant_message]
        return response, prompt + [assistant_message]
