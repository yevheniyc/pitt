import re
import time
from datetime import datetime

import pychrome
from bs4 import BeautifulSoup


def tab_start(tab):
    tab.start()
    tab.Page.enable()
    tab.DOM.enable()
    tab.Runtime.enable()


def create_and_navigate_to_calendar(browser):
    """Create a new tab and navigate to Google Calendar."""
    print("Creating new tab...")
    tab = browser.new_tab()
    tab_start(tab)
    print("Navigating to Google Calendar...")
    time.sleep(1)  # Small delay to ensure the tab is ready
    tab.Page.navigate(url="https://calendar.google.com/")
    return tab


def wait_and_scrape_html(tab, wait_time=4):
    """Wait for a fixed time, then scrape and return the full HTML of the page."""
    print(f"Waiting for {wait_time} seconds for the page to load...")
    time.sleep(wait_time)
    result = tab.Runtime.evaluate(expression="document.documentElement.outerHTML")
    if result and "result" in result and "value" in result["result"]:
        html = result["result"]["value"]
        print("Successfully retrieved HTML.")
        return html
    else:
        print("Failed to retrieve HTML.")
        return None


def clean_event_text(text):
    """Clean up event text by removing unnecessary information."""
    # Remove common prefixes and suffixes
    text = re.sub(r"Color: [^,]+,", "", text)
    text = re.sub(r"No location,", "", text)
    text = re.sub(r"Location: [^,]+,", "", text)
    text = re.sub(r"February \d+, \d+", "", text)
    text = re.sub(r"Accepted,", "", text)
    text = re.sub(r"Needs RSVP,", "", text)
    text = re.sub(r"Yevheniy Chuba,", "", text)
    # Remove multiple spaces and clean up
    text = re.sub(r"\s+", " ", text).strip()
    # Remove any remaining trailing commas and spaces
    text = re.sub(r",\s*$", "", text)
    return text


def extract_events_from_html(html):
    """
    Extract event-like information from the HTML.
    Uses BeautifulSoup to parse the calendar structure and find events.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Debug: print the structure we're working with
    print("\nPage structure:")
    print("Title:", soup.title.string if soup.title else "No title found")

    # Look for event containers
    event_containers = soup.find_all(
        ["div", "span"], attrs={"role": ["button", "gridcell"], "data-eventid": True}
    )

    events = []

    if not event_containers:
        print("\nLooking for alternative event markers...")
        # Try alternative selectors
        event_containers = soup.find_all(
            ["div", "span"],
            class_=lambda x: x
            and any(
                marker in str(x).lower()
                for marker in ["event", "calendar-container", "appointment"]
            ),
        )

    print(f"\nFound {len(event_containers)} potential event containers")

    for container in event_containers:
        # Get all text within this container
        text_content = container.get_text(separator=" ", strip=True)

        if not text_content or text_content in ["Home", "Change"]:
            continue

        # Debug: print container content
        print(f"\nProcessing: {text_content[:100]}")

        # Extract time range and event title
        time_range_pattern = re.compile(
            r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*(?:to|-|–)\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))"
        )
        time_match = time_range_pattern.search(text_content)

        if time_match:
            start_time, end_time = time_match.groups()
            # Get the text after the time range
            event_text = text_content[time_match.end() :].strip()
            event_text = clean_event_text(event_text)
            if event_text:
                events.append((f"{start_time} - {end_time}", event_text))
        else:
            # Check for all-day events
            event_text = clean_event_text(text_content)
            if event_text and not any(
                x in event_text.lower() for x in ["home", "change"]
            ):
                events.append(("All day", event_text))

    # Sort events by time
    def sort_key(event):
        time_str = event[0]
        if time_str == "All day":
            return (0, 0)  # All day events come first
        start_time = time_str.split(" - ")[0]
        try:
            # Convert to 24-hour format for sorting
            return (1, datetime.strptime(start_time, "%I:%M%p").time())
        except:
            try:
                return (1, datetime.strptime(start_time, "%I%p").time())
            except:
                return (2, 0)  # Put unparseable times at the end

    events.sort(key=sort_key)
    return events


def main():
    try:
        print("Connecting to Chrome...")
        browser = pychrome.Browser(url="http://localhost:9222")
        tab = create_and_navigate_to_calendar(browser)

        # Wait for a fixed time and then get the HTML content
        html = wait_and_scrape_html(tab, wait_time=4)
        if html is None:
            print("Could not retrieve HTML from the page.")
            return

        # Extract event details from the HTML
        events = extract_events_from_html(html)

        if events:
            print("\n=== Today's Schedule ===")
            current_date = None
            for time_str, event_detail in events:
                # Try to extract date if present
                date_match = re.search(r"February \d+", event_detail)
                if date_match and date_match.group() != current_date:
                    current_date = date_match.group()
                    print(f"\n{current_date}:")
                print(f"{time_str:15} | {event_detail}")
        else:
            print("\nNo events found for today.")

        # Save HTML for debugging (optional)
        with open("calendar_debug.html", "w", encoding="utf-8") as f:
            f.write(html)
            print("\nSaved HTML to calendar_debug.html for inspection")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
