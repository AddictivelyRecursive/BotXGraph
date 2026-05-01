from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureEncoder:
    """
    Lightweight text encoder for environments where transformer packages may not
    be installed. If sentence-transformers is available, it uses a MiniLM
    encoder; otherwise it falls back to TF-IDF + TruncatedSVD.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = int(embedding_dim)

    def encode(self, texts: Iterable[str]) -> Tuple[np.ndarray, str]:
        texts = ["" if text is None else str(text) for text in texts]
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), "empty"

        embeddings, source = self._try_sentence_transformer(texts)
        if embeddings is not None:
            return embeddings.astype(np.float32), source

        return self._tfidf_svd(texts), "tfidf_svd_fallback"

    def _try_sentence_transformer(self, texts: List[str]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None, None

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings, "sentence_transformer_all_MiniLM_L6_v2"

    def _tfidf_svd(self, texts: List[str]) -> np.ndarray:
        non_empty = [text for text in texts if text.strip()]
        if not non_empty:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=max(256, self.embedding_dim * 8),
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts)

        feature_dim = tfidf.shape[1]
        target_dim = min(self.embedding_dim, max(1, feature_dim - 1))
        if target_dim <= 1:
            dense = tfidf.toarray().astype(np.float32)
            if dense.shape[1] >= self.embedding_dim:
                return dense[:, :self.embedding_dim]

            out = np.zeros((dense.shape[0], self.embedding_dim), dtype=np.float32)
            out[:, :dense.shape[1]] = dense
            return out

        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        reduced = svd.fit_transform(tfidf).astype(np.float32)
        if reduced.shape[1] == self.embedding_dim:
            return reduced

        out = np.zeros((reduced.shape[0], self.embedding_dim), dtype=np.float32)
        out[:, :reduced.shape[1]] = reduced
        return out
