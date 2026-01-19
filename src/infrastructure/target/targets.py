import cocoindex
from cocoindex import op
from typing import Protocol

from src.infrastructure.target.chroma import ChromaTarget

class TargetStrategy(Protocol):
    """Target strategy interface"""
    def create(self, **kwargs) -> op.TargetSpec:
        ...

class QdrantTargetStrategy:
    """Qdrant target strategy"""
    def create(self, **kwargs) -> op.TargetSpec:
        return cocoindex.targets.Qdrant(
            connection=kwargs.get("connection"),
            collection_name=kwargs.get("collection_name"),
        )

class ChromaTargetStrategy:
    """Chroma target strategy"""
    def create(self, **kwargs) -> op.TargetSpec:
        return ChromaTarget(
            collection_name=kwargs.get("collection_name"),
            persist_directory=kwargs.get("persist_directory"),
            url=kwargs.get("url"),
        )

# Registry
TARGET_REGISTRY: dict[str, type[TargetStrategy]] = {
    "qdrant": QdrantTargetStrategy,
    "chroma": ChromaTargetStrategy,
}

def create_target(target_type: str, **kwargs) -> op.TargetSpec:
    """Create TargetSpec from registry with flexible parameters"""
    if target_type not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target type: {target_type}. Available: {list(TARGET_REGISTRY.keys())}")
    
    strategy_class = TARGET_REGISTRY[target_type]
    strategy = strategy_class()
    return strategy.create(**kwargs)