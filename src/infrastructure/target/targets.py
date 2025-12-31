import cocoindex
from cocoindex import op
from typing import Protocol

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

# Registry
TARGET_REGISTRY: dict[str, type[TargetStrategy]] = {
    "qdrant": QdrantTargetStrategy,
}

def create_target(target_type: str, **kwargs) -> op.TargetSpec:
    """Create TargetSpec from registry with flexible parameters"""
    if target_type not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target type: {target_type}. Available: {list(TARGET_REGISTRY.keys())}")
    
    strategy_class = TARGET_REGISTRY[target_type]
    strategy = strategy_class()
    return strategy.create(**kwargs)