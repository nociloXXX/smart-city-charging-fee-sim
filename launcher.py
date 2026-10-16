"""打包入口：默认启动桌面窗口界面；--web 用浏览器打开，--cli 跑命令行全量仿真。"""

import ctypes
import os
import sys


def _setup_console() -> None:
    """控制台按 UTF-8 输出，避免中文显示成乱码。"""
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _wait_for_enter() -> None:
    print("=" * 58)
    print("  面向智慧城市的快充负荷服务费定制策略仿真")
    print("=" * 58)
    input("按 Enter 键启动 ...")


def main() -> int:
    _setup_console()
    argv = sys.argv[1:]
    if "--cli" in argv:
        sys.argv = [sys.argv[0]] + [a for a in argv if a != "--cli"]
        from fcs_sim.main import main as cli_main

        return cli_main()

    interactive = (
        "--no-wait" not in argv
        and os.environ.get("FCSIM_NO_WAIT") != "1"
        and sys.stdin is not None
        and sys.stdin.isatty()
    )
    if interactive:
        try:
            _wait_for_enter()
        except (EOFError, KeyboardInterrupt):
            return 0

    print("加载中，请稍等 ...")
    sys.stdout.flush()

    if "--web" in argv:
        from fcs_sim.webgui import main as web_main

        return web_main()
    from fcs_sim.appgui import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
