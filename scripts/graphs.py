"""Every dashboard chart on its own: saved as PNGs in graphs/ (open the folder in VS Code; graphs/README.md shows them all
in the Markdown preview, Shift+Cmd+V) and, with --open, shown live in a window of their own.

    python scripts/graphs.py                    # rebuild the dashboard, then save graphs/*.png
    python scripts/graphs.py --open             # the same, then open the graphs window
    python scripts/graphs.py --no-save --open   # only open the window

The window is runs/dashboard.html?graphs in Chrome's app mode (no tabs or address bar): the dashboard page with only its
charts, which reloads itself every minute while a run is in progress, like the dashboard. The PNGs are a snapshot; re-run
to refresh them. Headless Chrome (the installed Google Chrome, driven over --remote-debugging-pipe, so no extra packages)
takes them from runs/dashboard.html?graphs&export in the light theme at 2x, one per card, and one per view for the cards
with several (the 3D view, the heat map). Metrics and aggregates only, like the dashboard.
"""
import argparse
import base64
import fcntl
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "runs/dashboard.html"
OUT = ROOT / "graphs"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WIDTH, SCALE, MAX_PX = 1280, 2, 16000  # page width (CSS px), PNG pixels per CSS px, longest PNG side to ask Chrome for

# the visible card's box in page coordinates, or null when it has nothing drawn
RECT = """(id => { const c = document.querySelector(`[data-graph="${id}"]`);
  if (!c || c.hidden || !c.querySelector("svg, table")) return null;
  const r = c.getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; })"""
CARDS = """[...document.querySelectorAll("#gallery [data-graph]")].filter(c => !c.hidden).map(c => ({ id: c.dataset.graph,
  title: c.dataset.title, section: c.dataset.section, variants: JSON.parse(c.dataset.variants || "null") }))"""


class Chrome:
    """Headless Chrome on the DevTools protocol over a pipe: commands go in on its fd 3, replies come out on fd 4, each
    message one JSON object ending in a NUL byte."""

    def __init__(self):
        self.profile = tempfile.mkdtemp(prefix="graphs-chrome-")
        cmd_r, self.cmd_w = os.pipe()
        self.out_r, out_w = os.pipe()

        def fds():  # in the child, before exec: the two pipe ends become fds 3 and 4 (copied high first, so neither clobbers the other)
            a, b = fcntl.fcntl(cmd_r, fcntl.F_DUPFD_CLOEXEC, 10), fcntl.fcntl(out_w, fcntl.F_DUPFD_CLOEXEC, 10)
            os.dup2(a, 3)
            os.dup2(b, 4)

        self.proc = subprocess.Popen(
            [str(CHROME), "--headless=new", "--remote-debugging-pipe", f"--user-data-dir={self.profile}", "--no-first-run",
             "--no-default-browser-check", "--hide-scrollbars", "about:blank"],
            preexec_fn=fds, close_fds=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(cmd_r)
        os.close(out_w)
        self.buf, self.n, self.events, self.session = bytearray(), 0, [], None
        target = self.call("Target.createTarget", url="about:blank")["targetId"]
        self.session = self.call("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]

    def read(self, timeout=60):
        while (end := self.buf.find(b"\0")) < 0:
            if not select.select([self.out_r], [], [], timeout)[0]:
                raise TimeoutError("no reply from Chrome")
            chunk = os.read(self.out_r, 1 << 20)
            if not chunk:
                raise RuntimeError("Chrome closed the DevTools pipe")
            self.buf += chunk
        msg = json.loads(self.buf[:end])
        del self.buf[:end + 1]
        return msg

    def call(self, method, **params):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params} | ({"sessionId": self.session} if self.session else {})
        data = json.dumps(msg).encode() + b"\0"
        while data:
            data = data[os.write(self.cmd_w, data):]
        while (m := self.read()).get("id") != self.n:
            self.events.append(m)
        if "error" in m:
            raise RuntimeError(f"{method}: {m['error'].get('message')}")
        return m.get("result", {})

    def wait(self, event, timeout=60):
        end = time.time() + timeout
        while True:
            for i, e in enumerate(self.events):
                if e.get("method") == event:
                    return self.events.pop(i)
            self.events.append(self.read(max(1, end - time.time())))

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page script failed: {r['exceptionDetails'].get('exception', {}).get('description', r['exceptionDetails'])}")
        return r["result"].get("value")

    def png(self, rect, path):
        scale = min(1.0, MAX_PX / (SCALE * max(rect["w"], rect["h"])))  # a very wide heat map comes out at a lower resolution
        clip = {"x": rect["x"], "y": rect["y"], "width": rect["w"], "height": rect["h"], "scale": scale}
        path.write_bytes(base64.b64decode(self.call("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)["data"]))

    def close(self):
        try:
            self.session = None
            self.call("Browser.close")
            self.proc.wait(10)
        except Exception:
            self.proc.kill()
        os.close(self.cmd_w)
        os.close(self.out_r)
        shutil.rmtree(self.profile, ignore_errors=True)


def save(out=OUT):
    """runs/dashboard.html?graphs&export -> out/NN-<card>[-<view>].png and out/README.md. Returns the number of PNGs."""
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        if re.match(r"\d\d-", f.name):
            f.unlink()  # last snapshot's charts, so one that is no longer drawn does not linger
    ch = Chrome()
    try:
        ch.call("Page.enable")
        ch.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)
        ch.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        ch.call("Page.navigate", url=PAGE.as_uri() + "?graphs&export")
        ch.wait("Page.loadEventFired")
        ref, built = ch.js("[DATA.reference, DATA.built]")
        lines, section, n = [], None, 0
        for i, card in enumerate(ch.js(CARDS), 1):
            shots = []
            for view, button in card["variants"] or [[None, None]]:
                if button:
                    ch.js(f"document.querySelector({json.dumps(button)}).click()")
                rect = ch.js(f"{RECT}({json.dumps(card['id'])})")
                if not rect:
                    continue  # this view has nothing to draw yet
                name = f"{i:02d}-{card['id']}" + (f"-{view}" if view else "") + ".png"
                ch.png(rect, out / name)
                shots.append((view, name))
            if not shots:
                continue
            if card["section"] != section:
                section = card["section"]
                lines += [f"## {section}", ""]
            lines += [f"### {card['title']}", ""]
            for view, name in shots:
                lines += ([f"*{view}*", ""] if view else []) + [f"![{card['title']}{' · ' + view if view else ''}]({name})", ""]
            n += len(shots)
    finally:
        ch.close()
    head = [f"# Knee results graphs", "",
            f"Saved {datetime.now():%Y-%m-%d %H:%M} by `scripts/graphs.py` from the dashboard built {built.replace('T', ' ')}, "
            f"reference run `{ref}`. A snapshot: run `start/5_graphs.command` (or `.venv/bin/python scripts/graphs.py`) to refresh. "
            "Metrics and aggregates only.", ""]
    (out / "README.md").write_text("\n".join(head + lines))
    return n


def open_window():
    url = PAGE.as_uri() + "?graphs"
    if CHROME.exists():  # app mode: a window with the page alone, no tabs or address bar
        subprocess.run(["open", "-na", "Google Chrome", "--args", f"--app={url}", "--window-size=1500,1000"], check=False)
    else:
        webbrowser.open(url)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--open", action="store_true", help="open the graphs window")
    p.add_argument("--no-save", action="store_true", help="do not save the PNGs")
    p.add_argument("--no-build", action="store_true", help="use runs/dashboard.html as it is")
    args = p.parse_args()
    if not args.no_build:
        subprocess.run([sys.executable, str(ROOT / "scripts/dashboard.py"), "--quiet"], check=False)
    if not PAGE.exists():
        sys.exit(f"{PAGE.relative_to(ROOT)} is missing: build it with scripts/dashboard.py")
    if not args.no_save:
        if not CHROME.exists():
            sys.exit(f"saving the PNGs needs Google Chrome at {CHROME}")
        t = time.time()
        n = save()
        print(f"saved {n} graphs to {OUT.relative_to(ROOT)}/ in {time.time() - t:.0f} s (all of them: {OUT.relative_to(ROOT)}/README.md)")
    if args.open:
        open_window()


if __name__ == "__main__":
    main()
