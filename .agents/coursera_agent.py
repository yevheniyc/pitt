import argparse
import json
import os
import time
from urllib.parse import urlparse

import pychrome
from bs4 import BeautifulSoup


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Coursera lecture scraper")
    parser.add_argument("--url", required=True, help="Coursera URL to scrape")
    parser.add_argument("--debug", action="store_true", help="Save debug HTML")
    return parser.parse_args()


def validate_coursera_url(url):
    """Validate that the URL is a Coursera URL."""
    parsed = urlparse(url)
    if parsed.netloc != "www.coursera.org":
        raise ValueError("Not a Coursera URL")
    if not parsed.path.startswith("/learn/"):
        raise ValueError("Not a Coursera course URL")
    return True


def tab_start(tab):
    tab.start()
    tab.call_method("Network.enable")
    tab.call_method("Page.enable")
    return tab


def create_and_navigate_to_coursera(browser, url):
    """Create a new tab and navigate to Coursera."""
    print("Creating new tab...")
    tab = browser.new_tab()
    tab = tab_start(tab)
    print(f"Navigating to {url}...")
    time.sleep(1)  # Allow the tab to initialize
    tab.call_method("Page.navigate", url=url)
    return tab


def wait_and_scrape_html(tab, wait_time=4):
    """Wait and get page HTML after JavaScript execution."""
    print(f"Waiting for {wait_time} seconds for the page to load...")
    time.sleep(wait_time)
    result = tab.call_method(
        "Runtime.evaluate", expression="document.documentElement.outerHTML"
    )
    html = result["result"]["value"]
    print("Successfully retrieved HTML.")
    return html


def check_login_status(html):
    soup = BeautifulSoup(html, "html.parser")
    # Check for login button or other indicators that user is not logged in
    login_button = soup.find("button", string="Log In")
    if login_button:
        print("\nYou need to be logged into Coursera to access lecture content.")
        print("Please log in using the opened Chrome window and try again.")
        return False
    return True


def extract_lecture_links(html):
    """Extract lecture links from a week/module page."""
    soup = BeautifulSoup(html, "html.parser")
    lectures = []

    # Try different selectors for lecture items
    items = (
        soup.find_all("div", {"class": "item"})
        or soup.find_all("div", {"data-test": "lesson-item"})
        or soup.find_all("a", href=lambda h: h and "/lecture/" in h)
    )

    for i, item in enumerate(items, 1):
        # If item is a link itself
        if item.name == "a":
            link = item
        else:
            # If item is a container, find the link inside
            link = item.find("a", href=lambda h: h and "/lecture/" in h)

        if not link:
            continue

        href = link.get("href")
        if not href:
            continue

        # Get title from link or its parent container
        title = link.get_text(strip=True)
        if not title and item.name != "a":
            title = item.get_text(strip=True)

        # Try to find duration
        duration = None
        duration_elem = item.find(
            ["span", "div"], string=lambda s: s and "min" in s.lower()
        )
        if duration_elem:
            duration = duration_elem.get_text(strip=True)

        # Clean up title and create URL
        title = title.split("min")[0].strip() if "min" in title else title
        url = f"https://www.coursera.org{href}" if href.startswith("/") else href

        lecture = {
            "number": i,
            "title": title,
            "duration": duration or "Unknown duration",
            "url": url,
        }
        lectures.append(lecture)

    if lectures:
        print(f"\nFound {len(lectures)} lectures:")
        for lecture in lectures:
            print(
                f"{lecture['number']}. {lecture['title']}•. Duration: {lecture['duration']}"
            )
            print(f"   {lecture['url']}")
    else:
        print("No lectures found on this page.")

    return lectures


def extract_transcript(html):
    soup = BeautifulSoup(html, "html.parser")

    # Look for transcript content in the first tab
    transcript_containers = [
        # Primary selectors for transcript tab content
        soup.find("div", {"role": "tabpanel"}),
        soup.find("div", {"data-testid": "transcript-panel-content"}),
        # Backup selectors
        soup.find("div", {"class": "rc-Transcript"}),
        soup.find("div", {"data-purpose": "transcript"}),
    ]

    transcript_container = next((c for c in transcript_containers if c), None)

    if not transcript_container:
        print("\nNo transcript found for this lecture")
        return None

    # Extract transcript text
    transcript_lines = []

    # Try to find transcript paragraphs
    for line in transcript_container.find_all(["p", "span"]):
        text = line.get_text(strip=True)
        # Skip metadata, instructions, and empty lines
        if (
            text
            and not text.startswith("Transcript language")
            and not text.startswith("Interactive Transcript")
            and not text.startswith("Play video")
            and not text.startswith("You may")
            and not text.startswith("For screen readers")
            and not text.isdigit()
            and not text.startswith("0:")
            and len(text) > 1
        ):  # Skip single characters
            transcript_lines.append(text)

    if not transcript_lines:
        print("\nTranscript container found but no text extracted")
        return None

    return "\n".join(transcript_lines)


def get_week_number(url):
    """Extract week/module number from URL."""
    path_parts = urlparse(url).path.split("/")
    if "week" in path_parts:
        week_idx = path_parts.index("week")
        return path_parts[week_idx + 1]
    elif "module" in path_parts:
        module_idx = path_parts.index("module")
        return path_parts[module_idx + 1]
    else:
        raise ValueError("URL must contain either 'week' or 'module'")


def process_lecture(browser, lecture_url, week_dir):
    """Process a single lecture and save its transcript."""
    print(f"\nProcessing lecture: {lecture_url}")

    tab = create_and_navigate_to_coursera(browser, lecture_url)
    html = wait_and_scrape_html(tab)

    if not check_login_status(html):
        return False

    transcript = extract_transcript(html)
    if transcript:
        # Extract lecture name for filename
        lecture_name = urlparse(lecture_url).path.split("/")[-1]
        filename = f"{lecture_name}.txt"
        output_file = os.path.join(week_dir, filename)

        # Save transcript with metadata
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Source: {lecture_url}\n\n")
            f.write(transcript)

        print(f"Saved transcript to: {output_file}")
        return True
    else:
        print("No transcript found for this lecture")
        return False


def process_week(url):
    """Process all lectures in a week/module."""
    try:
        # Extract week/module number and create directory
        period_num = get_week_number(url)
        period_type = "week" if "/home/week/" in url else "module"
        week_dir = os.path.join(
            "course_2", "scrapped_transcripts", f"{period_type}_{period_num}"
        )
        os.makedirs(week_dir, exist_ok=True)

        print(f"\nProcessing {period_type.title()} {period_num}")
        print(f"Saving transcripts to: {week_dir}")

        # Connect to Chrome
        print("\nConnecting to Chrome...")
        browser = pychrome.Browser(url="http://127.0.0.1:9222")

        # Get list of lectures
        tab = create_and_navigate_to_coursera(browser, url)
        html = wait_and_scrape_html(tab)

        if not check_login_status(html):
            return 1

        lectures = extract_lecture_links(html)
        if not lectures:
            print("No lectures found. Please check the URL and try again.")
            return 1

        # Process each lecture
        successful = 0
        failed = 0

        for lecture in lectures:
            try:
                if process_lecture(browser, lecture["url"], week_dir):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing lecture {lecture['url']}: {e}")
                failed += 1

        # Print summary
        print(f"\n{period_type.title()} {period_num} Processing Complete")
        print(f"Successfully processed: {successful} lectures")
        if failed > 0:
            print(f"Failed to process: {failed} lectures")

        return 0

    except Exception as e:
        print(f"An error occurred while processing week: {e}")
        return 1


def main():
    args = parse_args()

    try:
        validate_coursera_url(args.url)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if "/home/week/" not in args.url and "/home/module/" not in args.url:
        print(
            "Please provide a week or module URL (e.g., .../home/week/5 or .../home/module/1)"
        )
        return 1

    return process_week(args.url)


if __name__ == "__main__":
    exit(main())
