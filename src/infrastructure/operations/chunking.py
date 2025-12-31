import cocoindex
from cocoindex import op, functions
from typing import Protocol

class ChunkingStrategy(Protocol):
    """Chunking strategy interface"""
    def create(self, **kwargs) -> op.FunctionSpec:
        ...

class RecursiveChunkingStrategy:
    """Recursive chunking using SplitRecursively"""
    def create(self, **kwargs) -> op.FunctionSpec:
        return functions.SplitRecursively(
            custom_languages=kwargs.get("custom_languages", [
                functions.CustomLanguageSpec(
                    language_name="text",
                    separators_regex=[
                        r"\n(\s*\n)+",
                        r"[\.!\?]\s+",
                        r"\n",
                        r"\s+",
                    ],
                )
            ]),
            language=kwargs.get("language", "text"),
            chunk_size=kwargs.get("chunk_size", 600),
            chunk_overlap=kwargs.get("chunk_overlap", 100),
        )

class SeparatorChunkingStrategy:
    """Chunking using SplitBySeparators"""
    def create(self, **kwargs) -> op.FunctionSpec:
        return functions.SplitBySeparators(
            separators_regex=kwargs.get("separators_regex", [
                r"\n\n+",
                r"[\.!\?]\s+",
            ]),
            keep_separator=kwargs.get("keep_separator", "RIGHT"),
            include_empty=kwargs.get("include_empty", False),
            trim=kwargs.get("trim", True),
        )

# Registry
CHUNKING_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "recursive": RecursiveChunkingStrategy,
    "separator": SeparatorChunkingStrategy,
}

def get_chunking_function(method: str, **kwargs) -> op.FunctionSpec:
    """Get chunking FunctionSpec from registry with flexible parameters"""
    if method not in CHUNKING_REGISTRY:
        raise ValueError(f"Unknown chunking method: {method}. Available: {list(CHUNKING_REGISTRY.keys())}")
    
    strategy_class = CHUNKING_REGISTRY[method]
    strategy = strategy_class()
    return strategy.create(**kwargs)