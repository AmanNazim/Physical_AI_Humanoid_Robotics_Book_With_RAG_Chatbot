"""
Test script to process a single URL and identify where the hanging occurs
"""
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_single_url_processing():
    """Test processing of a single URL to identify hanging points"""
    logger.info("Testing single URL processing to identify hanging points...")

    # Step 1: Test URL fetching
    logger.info("Step 1: Testing URL fetching...")
    import aiohttp
    from bs4 import BeautifulSoup
    import re

    url = "https://amannazim.github.io/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/docs/preface/"

    start_time = asyncio.get_event_loop().time()
    async with aiohttp.ClientSession() as session:
        logger.info(f"Fetching: {url}")
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                html_content = await response.text()
                logger.info(f"✓ Fetched content in {asyncio.get_event_loop().time() - start_time:.2f}s")

                # Extract clean content
                soup = BeautifulSoup(html_content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()

                # Try to find the main content area
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'.*docMainContainer.*|.*main-wrapper.*|.*theme-doc-markdown.*|.*markdown.*'))

                if main_content:
                    content = main_content.get_text()
                else:
                    # Fallback to body if main content not found
                    body = soup.find('body')
                    if body:
                        content = body.get_text()
                    else:
                        content = soup.get_text()

                # Clean up the text
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                content = re.sub(r'\s+', ' ', content)

                logger.info(f"✓ Extracted clean content: {len(content)} characters")

    # Step 2: Test metadata creation
    logger.info("Step 2: Testing metadata creation...")
    metadata = {
        "document_reference": "Preface: Introduction to Physical AI & Humanoid Robotics",
        "page_reference": url,
        "section_title": "Preface: Introduction to Physical AI & Humanoid Robotics",
        "processing_version": "1.0",
        "additional_metadata": {
            "source_url": url,
            "content_type": "book_chapter",
            "language": "en",
            "module": "physical_ai_humanoid_robotics",
            "batch_group": "A_preface"
        }
    }
    logger.info("✓ Metadata created successfully")

    # Step 3: Test process_and_store function with timeout
    logger.info("Step 3: Testing process_and_store function with timeout...")
    start_time = asyncio.get_event_loop().time()

    try:
        from . import process_and_store
        # Run with timeout to catch hanging
        result = await asyncio.wait_for(
            process_and_store(content, metadata),
            timeout=120.0  # 2 minutes timeout
        )

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"✓ process_and_store completed in {elapsed:.2f}s")
        logger.info(f"Result: {result}")

        return True

    except asyncio.TimeoutError:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(f"✗ process_and_store timed out after {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(f"✗ process_and_store failed after {elapsed:.2f}s: {str(e)}")
        return False

async def main():
    success = await test_single_url_processing()
    if success:
        logger.info("🎉 Single URL processing test completed successfully!")
    else:
        logger.info("❌ Single URL processing test failed or timed out.")

if __name__ == "__main__":
    asyncio.run(main())