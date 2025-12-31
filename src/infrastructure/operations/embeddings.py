import cocoindex
import PIL
import io
import torch
import functools
from transformers import CLIPModel, CLIPProcessor


class TextEmbedder:
    """Text embedding using SentenceTransformer model"""
    
    def __init__(self, text_embedding_model_name: str | None = None):
        self.text_embedding_model_name = text_embedding_model_name
        
    def __call__(
        self,
        text: cocoindex.DataSlice[str],
    ) -> cocoindex.DataSlice[cocoindex.Vector[cocoindex.Float32]]:
        """Embed the text using a SentenceTransformer model"""
        return text.transform(
            cocoindex.functions.SentenceTransformerEmbed(
                model=self.text_embedding_model_name
            )
        )
        
class ImageEmbedder:
    """Image embedding using CLIP model"""
    _model_cache: dict[str, tuple[CLIPModel, CLIPProcessor]] = {}
    MODEL_NAME = "openai/clip-vit-large-patch14" # 하드 코딩
    
    def __init__(self, image_embedding_model_name: str | None = None):
        self.image_embedding_model_name = image_embedding_model_name or self.MODEL_NAME
        
    @functools.cache
    def _get_clip_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        """Get CLIP model"""
        model = CLIPModel.from_pretrained(self.image_embedding_model_name)
        processor = CLIPProcessor.from_pretrained(self.image_embedding_model_name)
        return model, processor
    
    @staticmethod
    def _get_clip_model_cached(model_name: str) -> tuple[CLIPModel, CLIPProcessor]:
        """Get CLIP model with caching"""
        if model_name not in ImageEmbedder._model_cache:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            ImageEmbedder._model_cache[model_name] = (model, processor)
        return ImageEmbedder._model_cache[model_name]
        
    def __call__(
        self,
        image_data: cocoindex.DataSlice[bytes],
    ) -> cocoindex.DataSlice[cocoindex.Vector[cocoindex.Float32]]:
        """Embed images using CLIP model"""
        return image_data.transform(embed_image_clip)

@cocoindex.op.function(cache=True, behavior_version=1, gpu=True)
def embed_image_clip(img_bytes: bytes) -> list[float]:
    """Embed a single image using CLIP model"""
    model_name = ImageEmbedder.MODEL_NAME
    model, processor = ImageEmbedder._get_clip_model_cached(model_name)
    image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].tolist()