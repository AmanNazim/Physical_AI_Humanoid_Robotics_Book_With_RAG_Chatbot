#!/usr/bin/env python3
"""
Script to parse logs and identify failed URLs from embedding pipeline
"""
import re
import sys
from typing import List, Dict, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_logs_for_failed_urls(log_file_path: str = None, log_content: str = None) -> Dict[str, List[str]]:
    """
    Parse logs to identify failed URLs based on error patterns.

    Args:
        log_file_path: Path to log file to parse
        log_content: Direct log content to parse (alternative to file path)

    Returns:
        Dictionary with lists of failed and successful URLs
    """
    if log_file_path:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    elif log_content:
        content = log_content
    else:
        raise ValueError("Either log_file_path or log_content must be provided")

    # Pattern to match URL processing lines
    processing_pattern = r'Processing URL \d+/\d+: (https?://[^\s]+)'
    # Pattern to match successful storage
    success_pattern = r'Successfully stored \d+ embeddings to database'
    # Pattern to match errors
    error_pattern = r'ERROR.*'
    # Pattern to match URL processing followed by error (without success)
    url_error_pattern = r'Processing URL \d+/\d+: (https?://[^\s]+).*?(?=Processing URL|\Z)'

    # Find all URLs being processed
    all_urls = re.findall(processing_pattern, content)

    # Find sections between URL processing lines to identify failures
    sections = re.split(r'Processing URL \d+/\d+: ', content)

    successful_urls = set()
    failed_urls = set()

    # Process each section to identify success/failure
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract URL from this section
        url_match = re.match(r'(https?://[^\s]+)', section)
        if not url_match:
            continue

        url = url_match.group(1)

        # Check if this URL section contains success
        if 'Successfully stored' in section:
            successful_urls.add(url)
        elif 'ERROR' in section or 'Failed to store' in section or 'timeout' in section.lower():
            # Check if this URL was fully processed or if it failed
            failed_urls.add(url)

    # Alternative approach: Look for URLs that were followed by errors before next URL
    lines = content.split('\n')
    current_url = None
    url_status = {}  # url -> status (success/failed)

    for line in lines:
        # Check if this line starts processing a new URL
        url_match = re.search(processing_pattern, line)
        if url_match:
            current_url = url_match.group(1)
            url_status[current_url] = 'processing'

        # Check for success after URL processing
        if current_url and 'Successfully stored' in line:
            url_status[current_url] = 'success'

        # Check for errors after URL processing
        if current_url and ('ERROR' in line or 'Failed to store' in line or 'timeout' in line.lower()):
            if url_status.get(current_url) != 'success':  # Only mark as failed if not already successful
                url_status[current_url] = 'failed'

    # Separate successful and failed based on status
    successful_urls = {url for url, status in url_status.items() if status == 'success'}
    failed_urls = {url for url, status in url_status.items() if status == 'failed'}

    # Also check for any URLs that might have failed due to timeout or other issues
    # by looking for patterns where a URL was processed but no success was logged
    all_processed_urls = set(all_urls)
    for url in all_processed_urls:
        if url not in successful_urls and url not in failed_urls:
            # Check the content around this URL to see if it failed
            url_pattern = f'Processing URL \\d+/\\d+: {re.escape(url)}'
            matches = list(re.finditer(url_pattern, content))
            for match in matches:
                # Find the content from this URL until the next URL or end of content
                start_pos = match.end()
                next_url_match = re.search(r'Processing URL \d+/\d+: https?://', content[start_pos:])
                if next_url_match:
                    end_pos = start_pos + next_url_match.start()
                    url_section = content[start_pos:end_pos]
                else:
                    url_section = content[start_pos:]

                # Check if this section contains success or error
                if 'Successfully stored' in url_section:
                    successful_urls.add(url)
                elif 'ERROR' in url_section or 'Failed to store' in url_section or 'timeout' in url_section.lower():
                    failed_urls.add(url)

    logger.info(f"Found {len(successful_urls)} successful URLs and {len(failed_urls)} failed URLs")

    return {
        'successful': sorted(list(successful_urls)),
        'failed': sorted(list(failed_urls)),
        'all_processed': sorted(list(all_processed_urls))
    }

def save_failed_urls_to_file(failed_urls: List[str], output_file: str = "failed_urls.txt"):
    """Save failed URLs to a file for retry processing."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in failed_urls:
            f.write(f"{url}\n")
    logger.info(f"Saved {len(failed_urls)} failed URLs to {output_file}")

def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_failed_urls.py <log_file_path> [output_file]")
        print("Or: python parse_failed_urls.py --stdin [output_file]")
        sys.exit(1)

    input_source = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "failed_urls.txt"

    if input_source == "--stdin":
        # Read from stdin
        log_content = sys.stdin.read()
        results = parse_logs_for_failed_urls(log_content=log_content)
    else:
        # Read from file
        results = parse_logs_for_failed_urls(log_file_path=input_source)

    print(f"Total URLs processed: {len(results['all_processed'])}")
    print(f"Successful URLs: {len(results['successful'])}")
    print(f"Failed URLs: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed URLs:")
        for url in results['failed']:
            print(f"  - {url}")

        save_failed_urls_to_file(results['failed'], output_file)
    else:
        print("\nNo failed URLs found!")