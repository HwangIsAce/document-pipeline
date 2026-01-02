from cocoindex import op, functions
from typing import Protocol

from src.infrastructure.chunkers.strategies.recursive import get_recursive_separators

class ChunkingStrategy(Protocol):
    """Chunking strategy interface"""
    def create(self, **kwargs) -> op.FunctionSpec:
        ...

class RecursiveChunkingStrategy:
    """Recursive chunking using SplitRecursively"""
    def create(self, **kwargs) -> op.FunctionSpec:
        default_separators = get_recursive_separators()
        
        return functions.SplitRecursively(
            custom_languages=kwargs.get("custom_languages", [
                functions.CustomLanguageSpec(
                    language_name="text",
                    separators_regex=default_separators,
                )
            ]),
        )

# Registry
CHUNKING_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "recursive": RecursiveChunkingStrategy,
}

def get_chunking_function(method: str, **kwargs) -> tuple[op.FunctionSpec, dict]:
    """Get chunking FunctionSpec and return unused kwargs for transform()"""
    if method not in CHUNKING_REGISTRY:
        raise ValueError(f"Unknown chunking method: {method}. Available: {list(CHUNKING_REGISTRY.keys())}")
    
    strategy_class = CHUNKING_REGISTRY[method]
    strategy = strategy_class()
    
    used_keys = {
        "recursive": {"custom_languages"},
    }
    
    used = used_keys.get(method, set())
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in used}
    
    return strategy.create(**kwargs), unused_kwargs