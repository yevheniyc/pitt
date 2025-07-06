import argparse
import re
import time
from urllib.parse import urlparse

import pychrome
from bs4 import BeautifulSoup


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Scrape content from Linear pages")
    parser.add_argument(
        "--url",
        type=str,
        default="https://linear.app/abridge/my-issues/assigned",
        help="URL of the Linear page to scrape",
    )
    return parser.parse_args()


def validate_linear_url(url):
    """Validate that the URL is a Linear URL."""
    parsed = urlparse(url)
    if not parsed.netloc.endswith("linear.app"):
        raise ValueError("URL must be a Linear URL (linear.app)")
    return url


def tab_start(tab):
    tab.start()
    tab.Page.enable()
    tab.DOM.enable()
    tab.Runtime.enable()


def create_and_navigate_to_linear(browser, url):
    """Create a new tab and navigate to the specified Linear URL."""
    print("Creating new tab...")
    tab = browser.new_tab()
    tab_start(tab)
    print(f"Navigating to {url}...")
    time.sleep(1)  # Allow the tab to initialize
    tab.Page.navigate(url=url)
    return tab


def wait_and_scrape_html(tab, wait_time=4):
    """
    Wait for a fixed time to allow the page to load, then
    retrieve the full HTML of the page.
    """
    print(f"Waiting for {wait_time} seconds for the page to load...")
    time.sleep(wait_time)

    # Check if the page indicates a logged-out state
    logged_out = tab.Runtime.evaluate(
        expression='document.querySelector(".logged-out") !== null'
    )
    if logged_out.get("result", {}).get("value"):
        print("Not logged in to Linear. Please log in first.")
        return None

    # Retrieve the entire HTML after JavaScript has executed
    result = tab.Runtime.evaluate(expression="document.documentElement.outerHTML")
    if result and "result" in result and "value" in result["result"]:
        html = result["result"]["value"]
        print("Successfully retrieved HTML.")
        return html
    else:
        print("Failed to retrieve HTML.")
        return None


def extract_text_from_html(html):
    """
    Use BeautifulSoup to parse the HTML and extract all text.
    Also attempt to identify the type of page and format accordingly.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try to determine if this is a single issue page or a list page
    issue_view = soup.find(
        lambda tag: tag.get("data-testid", "").startswith("issue-detail")
    )

    if issue_view:
        # This is a single issue page
        print("\n=== Single Issue Details ===")

        # Try to get issue details
        title = soup.find(
            lambda tag: tag.get("data-testid", "").startswith("issue-title")
        )
        status = soup.find(
            lambda tag: tag.get("data-testid", "").startswith("issue-status")
        )
        priority = soup.find(
            lambda tag: tag.get("data-testid", "").startswith("issue-priority")
        )
        description = soup.find(
            lambda tag: tag.get("data-testid", "").startswith("issue-description")
        )

        if title:
            print(f"\nTitle: {title.get_text(strip=True)}")
        if status:
            print(f"Status: {status.get_text(strip=True)}")
        if priority:
            print(f"Priority: {priority.get_text(strip=True)}")
        if description:
            print(f"\nDescription:\n{description.get_text(strip=True)}")

    else:
        # This is probably a list page
        print("\n=== Issue List ===")
        text = soup.get_text(separator=" ", strip=True)

        # Try to extract issue IDs and titles using regex
        issue_pattern = re.compile(
            r"(DAT-\d+)\s+([^0-9\n]{3,}?)(?=DAT-|\d{1,2}\s*/\s*\d{1,2}|$)"
        )
        matches = issue_pattern.finditer(text)

        issues = []
        seen = set()  # To avoid duplicates
        for match in matches:
            issue_id = match.group(1)
            if issue_id not in seen:
                seen.add(issue_id)
                title = match.group(2).strip()
                issues.append((issue_id, title))

        if issues:
            print("\nFound Issues:")
            for issue_id, title in issues:
                print(f"- {issue_id}: {title}")
            print(f"\nTotal issues found: {len(issues)}")
        else:
            print("\nNo issues found. Raw text preview (first 1000 characters):")
            print(text[:1000])

    return soup.get_text(separator=" ", strip=True)


def main():
    try:
        args = parse_args()
        url = validate_linear_url(args.url)

        print("Connecting to Chrome...")
        browser = pychrome.Browser(url="http://localhost:9222")
        tab = create_and_navigate_to_linear(browser, url)

        html = wait_and_scrape_html(tab, wait_time=4)
        if html is None:
            print("Could not retrieve HTML from the page.")
            return

        # Extract and format text based on the page type
        extract_text_from_html(html)

        # Save the full HTML for inspection
        with open("linear_debug.html", "w", encoding="utf-8") as f:
            f.write(html)
            print("\nSaved HTML to linear_debug.html for inspection")

    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
