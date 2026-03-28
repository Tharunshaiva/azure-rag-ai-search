# shared/chunker.py

from venv import logger


class Chunker:
    def __init__(self, chunk_size: int = 1024, overlap: int = 256):
        """
        chunk_size: number of characters per chunk
        overlap: amount of characters shared between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        """
        Splits text into overlapping chunks.
        Returns a list of chunk strings.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        step = self.chunk_size - self.overlap  # e.g., 1024 - 256 = 768

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            start += step  # move by step, not full chunk size
        logger.info(f"chunks create successfully: {chunks[:]}")
        return chunks
