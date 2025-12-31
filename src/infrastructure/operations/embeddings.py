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
    def __init__(self, image_embedding_model_name: str | None = None):
        self.image_embedding_model_name = image_embedding_model_name
        
    @functools.cache
    def _get_clip_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        model = CLIPModel.from_pretrained(self.image_embedding_model_name)
        processor = CLIPProcessor.from_pretrained(self.image_embedding_model_name)
        return model, processor
        
    @cocoindex.op.function(cache=True, behavior_version=1, gpu=True)
    def __call__(self, img_bytes: bytes) -> list[float]:
        model, processor = self._get_clip_model()
        image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].tolist()
        
