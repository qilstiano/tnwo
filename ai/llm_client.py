import json
import urllib.request

# ============================================================
# OLLAMA CLIENT
# ============================================================
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"]

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

# ============================================================
# VLLM CLIENT  (OpenAI-compatible API served by vLLM)
# ============================================================
class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "your-model-name"):
        self.base_url = base_url
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 512,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False


def create_llm_client(provider: str, base_url: str, model: str):
    """Factory: build the right client based on provider type."""
    if provider == "vllm":
        return VLLMClient(base_url=base_url, model=model)
    elif provider == "ollama":
        return OllamaClient(base_url=base_url, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
