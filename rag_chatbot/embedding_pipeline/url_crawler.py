"""
URL-based document crawler for sitemap.xml
"""
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)


class URLCrawler:
    """Class for crawling URLs from a sitemap.xml file"""

    def __init__(self):
        self.session = None

    async def get_all_urls_from_sitemap(self, sitemap_url: str) -> List[Dict[str, Any]]:
        """Get all URLs from the specified sitemap URL"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(sitemap_url) as response:
                response.raise_for_status()
                sitemap_content = await response.text()
            return await self.parse_sitemap(sitemap_content)
        except Exception as e:
            logger.error(f"Failed to fetch or parse sitemap from {sitemap_url}: {str(e)}")
            raise

    async def parse_sitemap(self, sitemap_content: str) -> List[Dict[str, Any]]:
        """Parse the sitemap XML and extract URLs"""
        urls = []
        try:
            root = ET.fromstring(sitemap_content)

            # Handle regular sitemap with URLs
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None:
                    url_data = {
                        'url': loc_elem.text,
                        'lastmod': None,
                        'changefreq': None,
                        'priority': None
                    }

                    # Also look for other attributes
                    lastmod_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                    changefreq_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq')
                    priority_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}priority')

                    if lastmod_elem is not None:
                        url_data['lastmod'] = lastmod_elem.text
                    if changefreq_elem is not None:
                        url_data['changefreq'] = changefreq_elem.text
                    if priority_elem is not None:
                        url_data['priority'] = priority_elem.text

                    urls.append(url_data)
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {str(e)}")
            raise

        return urls

    async def fetch_content_from_url(self, url: str) -> str:
        """Fetch content from a specific URL"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url) as response:
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
            logger.error(f"Failed to fetch content from {url}: {str(e)}")
            raise

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


class SitemapParser:
    """Class for parsing sitemap.xml content"""

    @staticmethod
    def extract_urls_from_xml(sitemap_xml: str) -> List[str]:
        """Extract URLs from sitemap XML content"""
        urls = []
        try:
            root = ET.fromstring(sitemap_xml)

            # Regular sitemap with URLs
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None:
                    urls.append(loc_elem.text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {str(e)}")
            raise

        return urls


class URLProcessor:
    """Class for processing individual URLs for one-by-one processing"""

    def __init__(self, url: str):
        self.url = url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_content(self) -> str:
        """Fetch content from the URL"""
        try:
            async with self.session.get(self.url) as response:
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
            logger.error(f"Failed to fetch content from {self.url}: {str(e)}")
            raise