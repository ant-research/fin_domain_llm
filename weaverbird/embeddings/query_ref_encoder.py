from typing import List

from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch


class QueryRefEncoder(Embeddings):
    """
    Produce embeddings of query and references using a pretrained encoder
    """

    def __init__(self, model_dir, device=None, max_batch_size=400):
        super(QueryRefEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.query_encoder = AutoModel.from_pretrained(model_dir + "/query_encoder")
        self.reference_encoder = AutoModel.from_pretrained(model_dir + "/reference_encoder")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.query_encoder = self.query_encoder.to(self.device).eval()
        self.reference_encoder = self.reference_encoder.to(self.device).eval()
        assert max_batch_size > 0
        self.max_batch_size = max_batch_size

    def get_embeddings(self, sentences: List[str], encoder) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True,
                                    truncation=True, return_tensors='pt')
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            outputs = encoder(**inputs)
            # Mean Pool
            token_embeddings = outputs[0]
            mask = inputs["attention_mask"]
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(
                dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a trained retriever model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.get_embeddings(texts, self.reference_encoder)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a trained retriever model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.get_embeddings([text], self.query_encoder)[0]
        return embedding.tolist()
