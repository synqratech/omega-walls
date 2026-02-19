"""LLM backends for Omega RAG harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any, Dict


@dataclass
class LocalTransformersLLM:
    model_path: str = "."
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("transformers/torch are required for LocalTransformersLLM") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt: str) -> Dict[str, Any]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a careful enterprise assistant."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = prompt

        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": do_sample,
            "temperature": float(self.temperature) if do_sample else 1.0,
            "top_p": float(self.top_p) if do_sample else 1.0,
            "top_k": 50 if not do_sample else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with self._torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text.strip(),
            "backend": "local_transformers",
            "device": self.device,
        }


@dataclass
class OllamaLLM:
    model: str = "qwen:0.5b"
    endpoint: str = "http://localhost:11434/api/generate"
    timeout_seconds: int = 120

    def generate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON response") from exc

        text = str(data.get("response", "")).strip()
        return {
            "text": text,
            "backend": "ollama",
            "model": self.model,
        }
