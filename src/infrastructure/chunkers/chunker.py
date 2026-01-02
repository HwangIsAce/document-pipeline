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

def get_chunking_function(method: str, **kwargs) -> tuple[op.FunctionSpec, dict]:
    """Get chunking FunctionSpec and return unused kwargs for transform()"""
    if method not in CHUNKING_REGISTRY:
        raise ValueError(f"Unknown chunking method: {method}. Available: {list(CHUNKING_REGISTRY.keys())}")
    
    strategy_class = CHUNKING_REGISTRY[method]
    strategy = strategy_class()
    
    used_keys = {
        "recursive": {"custom_languages"},
        "separator": {"separators_regex", "keep_separator", "include_empty", "trim"},
    }
    
    used = used_keys.get(method, set())
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in used}
    
    return strategy.create(**kwargs), unused_kwargs