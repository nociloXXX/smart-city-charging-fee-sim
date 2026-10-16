"""桌面窗口界面：优先用内嵌窗口(pywebview)，否则用 Edge/Chrome 应用模式，
都不显示浏览器标签页与地址栏。"""

from __future__ import annotations

import subprocess
import threading
import webbrowser
from pathlib import Path

from .webgui import OUTDIR, Handler, LocalHTTPServer

DEFAULT_PORT = 8000

# Edge / Chrome 应用模式：独立窗口，没有标签页和地址栏
_BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _start_server(port: int) -> LocalHTTPServer:
    server = LocalHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(port: int = DEFAULT_PORT) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    server = _start_server(port)
    url = f"http://127.0.0.1:{port}/"
    embedded = False
    try:
        import webview

        webview.create_window(
            "快充负荷服务费定制策略仿真", url, width=1180, height=860, min_size=(900, 640)
        )
        webview.start()
        embedded = True
    except Exception:
        embedded = False

    if embedded:
        mode = "内嵌窗口"
    else:
        mode = "浏览器"
        for exe in _BROWSER_CANDIDATES:
            if Path(exe).is_file():
                subprocess.Popen([exe, f"--app={url}", "--window-size=1200,880"])
                mode = "应用窗口"
                break
        else:
            webbrowser.open(url)

    print(f"界面已启动（{mode}）：{url}")
    print("使用期间请不要关闭本窗口；关闭本窗口即退出程序。")
    try:
        if not embedded:
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
