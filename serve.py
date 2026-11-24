"""Start the ClearView web app:  python serve.py [--port 8000] [--host 127.0.0.1]"""
import argparse
import webbrowser

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ClearView web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        webbrowser.open(f"http://{a.host}:{a.port}")
    uvicorn.run("api:app", host=a.host, port=a.port, log_level="info")
