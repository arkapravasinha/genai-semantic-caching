from redisvl.utils.vectorize.base import BaseVectorizer
import os
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.utils import deprecated_argument
from google import genai
from google.genai import types


class GoogleGenAIVectorizer(BaseVectorizer):
    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype)
        # Init client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after init
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):

        try:

            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        except ImportError:
            raise ImportError(
                "GoogleGenAIVectorizer requires the `google-genai` package to be installed."
            )

    def _set_model_dims(self) -> int:
        try:
            embedding = self.embed("dimension check")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the VertexAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._client.models.embed_content(model=self.model, 
                                                   contents=batch,
                                                   config=types.EmbedContentConfig(output_dimensionality=768))
            embeddings += [
                self._process_embedding(r.values, as_buffer, dtype) for r in response.embeddings
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        print(f"\nEmbedding single text: '{text}'")
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = self._client.models.embed_content(model=self.model, 
                                                   contents=[text],
                                                   config=types.EmbedContentConfig(output_dimensionality=768))
        
        return self._process_embedding(result.embeddings[0].values, as_buffer, dtype)

    @property
    def type(self) -> str:
        return "google-genai"

# Testing the vectorizer
# goog= GoogleGenAIVectorizer(model="text-embedding-004")
# response=goog.embed_many(["hello world","how are you"])
# print(response)
