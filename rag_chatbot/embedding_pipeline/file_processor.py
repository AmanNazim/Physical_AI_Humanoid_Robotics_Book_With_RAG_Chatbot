"""
File processor for document ingestion
"""
import asyncio
import aiohttp
from pathlib import Path
from typing import Union, List, Optional
import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class FileProcessor:
    """Class for document ingestion and file processing"""

    def __init__(self):
        self.supported_formats = {'.txt', '.md', '.html', '.htm'}

    async def load_file(self, source: Union[str, Path]) -> str:
        """Load content from a file or URL"""
        source_path = Path(source)

        if source_path.suffix.lower() in self.supported_formats:
            # Load from local file
            return await self._load_local_file(source_path)
        elif self._is_url(str(source)):
            # Load from URL
            return await self._load_from_url(str(source))
        else:
            raise ValueError(f"Unsupported file format: {source_path.suffix}")

    async def _load_local_file(self, file_path: Path) -> str:
        """Load content from a local file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to load local file {file_path}: {str(e)}")
            raise

    async def _load_from_url(self, url: str) -> str:
        """Load content from a URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()

                    # Extract text content from HTML
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    return text
        except Exception as e:
            logger.error(f"Failed to load content from URL {url}: {str(e)}")
            raise

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def process_document_batch(self, sources: List[Union[str, Path]]) -> List[dict]:
        """Process a batch of documents"""
        results = []
        for source in sources:
            try:
                content = await self.load_file(source)
                results.append({
                    "source": str(source),
                    "content": content,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Failed to process document {source}: {str(e)}")
                results.append({
                    "source": str(source),
                    "content": "",
                    "success": False,
                    "error": str(e)
                })

        return results