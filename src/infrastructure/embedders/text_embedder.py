import cocoindex

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