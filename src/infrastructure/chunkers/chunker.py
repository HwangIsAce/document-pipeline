from cocoindex import op, functions
from typing import Protocol

from src.infrastructure.chunkers.strategies.llm_semantic import llm_semantic_chunk
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
        
class LLMChunkingStrategy:
    """LLM semantic chunking using llm_semantic_chunk"""
    def create(self, **kwargs) -> op.FunctionSpec:
        return llm_semantic_chunk

# Registry
CHUNKING_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "recursive": RecursiveChunkingStrategy,
    "llm_semantic": LLMChunkingStrategy,
}

def get_chunking_function(method: str, **kwargs) -> tuple[op.FunctionSpec, dict]:
    """Get chunking FunctionSpec and return unused kwargs for transform()"""
    if method not in CHUNKING_REGISTRY:
        raise ValueError(f"Unknown chunking method: {method}. Available: {list(CHUNKING_REGISTRY.keys())}")
    
    strategy_class = CHUNKING_REGISTRY[method]
    strategy = strategy_class()
    
    used_keys = {
        "recursive": {"custom_languages"},
        "llm_semantic": {"organization", "api_key", "model_name", "chunk_size", "chunk_overlap"},
    }
    
    used = used_keys.get(method, set())
    
    if method == "llm_semantic":
        func_spec = strategy.create(**kwargs)
        
        unused_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in used and v is not None
        }
        return func_spec, unused_kwargs
    
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in used}
    
    return strategy.create(**kwargs), unused_kwargs