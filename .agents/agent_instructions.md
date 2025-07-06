# Browser Automation Agents

This directory contains browser automation agents for various services using Python and `pychrome`.

## Prerequisites

1. Chrome/Chromium browser
2. Python 3.7+
3. Required packages (install via `pip install -r requirements.txt`):
   - pychrome>=0.2.3
   - beautifulsoup4>=4.12.0
   - requests>=2.31.0

## Chrome Setup

1. Start Chrome with remote debugging enabled in a separate terminal session:

   ```bash
   # On macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

   # On Linux
   google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

   # On Windows
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=C:\temp\chrome-debug
   ```

   Note: Using a separate user data directory (`--user-data-dir`) ensures a clean browser session without conflicts with your regular Chrome profile.

2. Important: Make sure you're logged into the services you want to access:
   - Coursera
   - Linear
   - Any other service you plan to automate

## Available Agents

### 1. Coursera Agent (`coursera_agent.py`)

Scrapes lecture transcripts from Coursera courses:

- Extracts lecture links from a week/module page
- Navigates to each lecture and extracts the transcript
- Saves transcripts to organized directories

Usage:

```bash
# After starting Chrome with remote debugging, in a separate terminal session:
python .agents/coursera_agent.py --url https://www.coursera.org/learn/mds-introduction-to-data-centric-computing/home/week/12
```

### 2. Linear Agent (`linear_agent.py`)

Automates Linear task management:

- View assigned tasks
- Get detailed task information
- Extract task requirements
- Future capabilities:
  - Task status updates
  - Comment management
  - Priority management

Usage:

```bash
# View assigned tasks
python linear_agent.py

# View specific task
python linear_agent.py --url https://linear.app/your-org/issue/TASK-ID
```

## Common Issues

1. **Chrome Connection Error**

   - Always use separate terminal sessions for Chrome and the agent script
   - Ensure Chrome is running with remote debugging enabled on port 9222
   - Include the `--user-data-dir=/tmp/chrome-debug` flag to avoid conflicts
   - If you still get connection errors, try killing all Chrome processes with `pkill -f "Google Chrome"` and starting fresh
   - Verify the connection is working with: `curl -s http://localhost:9222/json/version`

2. **Authentication Issues**

   - Make sure you're logged into the services in Chrome
   - Some services may require re-authentication
   - For Coursera: Log in to your account in the opened Chrome window

3. **Parsing Issues**
   - HTML structure might change with service updates
   - Check debug output with `--debug` flag
   - For Coursera: Make sure you have access to the course content

## Development

The agents use a common approach:

1. Connect to Chrome using `pychrome`
2. Navigate to the target URL
3. Wait for page load
4. Extract content using BeautifulSoup
5. Parse and format the output

To add new capabilities:

1. Update the relevant agent file
2. Add new command-line arguments if needed
3. Update this documentation
4. Test with different scenarios

## Files

- `agent_instructions.md`: This documentation
- `linear_agent.py`: Linear task management automation
- `requirements.txt`: Required Python packages
