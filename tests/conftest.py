import hashlib


class FakeEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest] * 12
        return values[: self.dim]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
