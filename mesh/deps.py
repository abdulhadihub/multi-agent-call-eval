from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mesh.config import Settings
    from mesh.router import ModelRouter
    from stores.chroma import ChromaMemory
    from stores.embeddings import Embedder
    from stores.sqlite import SqliteStore
    from tools.gateway import ToolGateway
    from tools.simulated_apis import SimulatedCompanyAPI


@dataclass
class MeshDeps:
    router: "ModelRouter"
    gateway: "ToolGateway"
    store: "SqliteStore"
    memory: "ChromaMemory"
    embedder: "Embedder"
    api: "SimulatedCompanyAPI"
    settings: "Settings"
    now: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc)
    )
