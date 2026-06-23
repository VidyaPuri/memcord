"""Non-blocking wrapper around sentence-transformers for async use.

SentenceTransformer.encode() is synchronous and blocks the asyncio event loop.
In production, this means every FAQ cache check freezes ALL Discord event processing
for 5-50ms. This wrapper offloads encoding to a thread pool.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging

log = logging.getLogger("memcord.embedding")


class AsyncEmbedder:
    """Runs sentence-transformers encoding in a thread pool.

    Usage:
        embedder = AsyncEmbedder("all-MiniLM-L6-v2", max_workers=2)
        embedding = await embedder.encode("hello world")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_workers: int = 2):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        log.info("AsyncEmbedder: %s loaded, %d workers", model_name, max_workers)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts asynchronously, returns list of embedding vectors."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            functools.partial(self._model.encode, texts, batch_size=32),
        )
        return result.tolist()

    async def encode_single(self, text: str) -> list[float]:
        """Encode a single text, returns one embedding vector."""
        results = await self.encode([text])
        return results[0]

    def close(self):
        """Shut down the thread pool."""
        self._executor.shutdown(wait=True)
