from __future__ import annotations

from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        supported = {
            item["model"]: item for item in TextEmbedding.list_supported_models()
        }
        if model_name not in supported:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        self.model_name = model_name
        self.dim = int(supported[model_name]["dim"])
        if self.dim != 384:
            raise ValueError(
                f"{model_name} produces {self.dim} dimensions, expected 384"
            )
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]
