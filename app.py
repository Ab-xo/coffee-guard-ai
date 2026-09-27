"""
CoffeeGuard AI - Hugging Face Space Entry Point
Ethiopian Coffee Leaf Disease Detection System
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        "apps/web/streamlit_app.py",
        "--server.headless=true",
        "--browser.serverAddress=0.0.0.0",
        "--server.enableCORS=false"
    ]
    sys.exit(stcli.main())
