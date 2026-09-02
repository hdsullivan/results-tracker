"""Screenshot Streamlit pages with headless Chrome over the DevTools protocol.

Usage: python scripts/screenshot.py http://localhost:8501 docs/screenshots [page ...]
Needs Google Chrome and `pip install websocket-client` (dev-only, not a package dependency).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from websocket import create_connection

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = ["overview", "comparison", "sweep", "ablation", "run", "export"]


def shoot(ws_url: str, url: str, out: Path, wait: float, width: int, height: int, full_page: bool = True) -> None:
    ws = create_connection(ws_url, timeout=60)
    mid = 0

    def call(method: str, **params):
        nonlocal mid
        mid += 1
        ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=False)
    call("Page.enable")
    call("Page.navigate", url=url)
    time.sleep(wait)
    if full_page:
        # grow the viewport to the document height so the whole page is captured
        # Streamlit scrolls inside its main container, so measure that rather than the document
        expr = ("(() => { const m = document.querySelector('[data-testid=\"stMain\"]') || document.documentElement;"
                " return Math.max(m.scrollHeight, document.documentElement.scrollHeight); })()")
        h = call("Runtime.evaluate", expression=expr, returnByValue=True)["result"]["value"]
        call("Emulation.setDeviceMetricsOverride", width=width, height=max(height, min(int(h) + 40, 2400)), deviceScaleFactor=2, mobile=False)
        time.sleep(1.5)
    shot = call("Page.captureScreenshot", format="png")
    out.write_bytes(base64.b64decode(shot["data"]))
    ws.close()


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots")
    pages = sys.argv[3:] or PAGES
    outdir.mkdir(parents=True, exist_ok=True)
    port = 9333
    chrome = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}", "--remote-allow-origins=*",
         "--user-data-dir=/tmp/rt-chrome-profile", "--window-size=1400,900", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                targets = json.loads(urllib.request.urlopen(f"http://localhost:{port}/json").read())
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("chrome did not start")
        ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        for page in pages:
            out = outdir / f"{page}.png"
            url = base if page == "overview" else f"{base}/{page}"  # the default page is served at the root
            shoot(ws_url, url, out, wait=7.0, width=1400, height=900)
            print("wrote", out)
    finally:
        chrome.terminate()


if __name__ == "__main__":
    main()
