"""本地网页界面：浏览器填参数 -> 运行仿真 -> 查看结果图。"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


os.environ.setdefault("MPLCONFIGDIR", str(_base_dir() / ".mplcache"))

from .config import Config, make_fee_levels
from .main import run_with

OUTDIR = _base_dir() / "results"


class LocalHTTPServer(ThreadingHTTPServer):
    """本地服务：绑定地址时跳过主机名反查，避免离线或 DNS 慢时卡住。"""

    allow_reuse_address = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port

# (显示名, 参数名, 类型)
FIELDS = [
    ("电动汽车数量", "n_ev", int),
    ("公共快充比例", "public_share", float),
    ("电池容量 kWh", "battery", float),
    ("触发 SOC 阈值", "soc_trigger", float),
    ("单桩功率 kW", "pile_power_kw", float),
    ("每站桩数", "piles_per_station", int),
    ("服务费下限 元/kWh", "fee_min", float),
    ("服务费上限 元/kWh", "fee_max", float),
    ("基础负荷缩放", "base_load_scale", float),
    ("电压下限 p.u.", "v_lower", float),
    ("最大迭代轮数", "max_iter", int),
    ("随机种子", "seed", int),
    ("路口间距 km", "spacing_km", float),
]


def _defaults() -> dict:
    cfg = Config()
    return {
        "n_ev": cfg.n_ev,
        "public_share": cfg.public_share,
        "battery": cfg.battery_range[0],
        "soc_trigger": cfg.soc_trigger,
        "pile_power_kw": cfg.pile_power_kw,
        "piles_per_station": cfg.piles_per_station,
        "fee_min": cfg.fee_levels[0],
        "fee_max": cfg.fee_levels[-1],
        "base_load_scale": cfg.base_load_scale,
        "v_lower": cfg.v_lower,
        "max_iter": cfg.max_iter,
        "seed": cfg.seed,
        "spacing_km": cfg.spacing_km,
    }


def _parse_params(raw: dict) -> dict:
    params = {}
    for _, key, typ in FIELDS:
        text = str(raw.get(key, "")).strip()
        if text == "":
            continue
        params[key] = typ(text)
    if "battery" in params:
        params["battery_range"] = (params.pop("battery"),) * 2
    fee_min = params.pop("fee_min", Config().fee_levels[0])
    fee_max = params.pop("fee_max", Config().fee_levels[-1])
    params["fee_levels"] = make_fee_levels(float(fee_min), float(fee_max))
    params["fee_init"] = round((float(fee_min) + float(fee_max)) / 2, 2)
    return params


def _html_page() -> str:
    defaults = _defaults()
    rows = []
    for idx, (label, key, _typ) in enumerate(FIELDS):
        rows.append(
            f'<label>{label}<input name="{key}" value="{defaults[key]}"></label>'
        )
    return PAGE_TEMPLATE.replace("__FIELDS__", "\n".join(rows))


PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>快充负荷服务费定制策略仿真</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", system-ui, sans-serif; background: #f4f6f8; color: #1f2937; }
  header { padding: 18px 28px; background: #1f2937; color: #fff; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header p { margin: 4px 0 0; font-size: 12px; color: #cbd5e1; }
  .warn { color: #fbbf24; font-weight: 600; margin-top: 8px; font-size: 13px; }
  main { max-width: 980px; margin: 24px auto; padding: 0 20px 60px; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 16px; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  label { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13px; }
  input { width: 150px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }
  button { padding: 9px 22px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; font-size: 14px; cursor: pointer; }
  button:disabled { background: #94a3b8; cursor: wait; }
  #status { margin-left: 12px; font-size: 13px; color: #475569; }
  pre { background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; overflow: auto; font-size: 12px; line-height: 1.5; }
  .figs { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 720px) { .figs { grid-template-columns: 1fr; } }
  figure { margin: 0; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
  figcaption { font-size: 12px; color: #475569; padding-top: 6px; }
  img { width: 100%; border-radius: 6px; }
</style>
</head>
<body>
<header>
  <h1>快充负荷服务费定制策略仿真</h1>
  <p>路网交通 -> 充电负荷 -> 配电网潮流 -> 服务费调整，迭代闭环</p>
  <p class="warn">使用期间请不要关闭软件（保持后台命令行窗口开启）</p>
</header>
<main>
  <section class="panel">
    <div class="grid">__FIELDS__</div>
    <div style="margin-top:16px;display:flex;align-items:center;">
      <button id="run" onclick="run()">运行仿真</button>
      <span id="status">就绪</span>
    </div>
    <p style="margin:10px 0 0;font-size:12px;color:#64748b;">
      相同参数与随机种子下结果可完全复现；想看到不同曲线，请改动参数或换一个随机种子。
    </p>
  </section>
  <section id="result" style="display:none;">
    <pre id="summary"></pre>
    <div class="figs" id="figs"></div>
  </section>
</main>
<script>
async function run() {
  const btn = document.getElementById("run");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "运行中，请稍候…";
  const payload = {};
  document.querySelectorAll("input").forEach(el => { payload[el.name] = el.value; });
  try {
    const resp = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    document.getElementById("result").style.display = "block";
    document.getElementById("summary").textContent = data.summary;
    const figs = document.getElementById("figs");
    figs.innerHTML = "";
    data.figures.forEach(f => {
      const fig = document.createElement("figure");
      const url = `/results/${f}?v=${Date.now()}`;
      fig.innerHTML = `<img src="${url}" alt="${f}"><figcaption>${f}</figcaption>`;
      figs.appendChild(fig);
    });
    status.textContent = "完成，结果已写入 results 目录";
  } catch (err) {
    status.textContent = "运行失败：" + err.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # 静默日志
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, _html_page().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.startswith("/results/"):
            name = path.split("/", 2)[-1].split("?")[0]
            if "/" in name or "\\" in name:
                self._send(400, b"bad path", "text/plain")
                return
            fp = OUTDIR / name
            if not fp.is_file():
                self._send(404, b"not found", "text/plain")
                return
            body = fp.read_bytes()
            ctype = "image/png" if fp.suffix == ".png" else "text/plain; charset=utf-8"
            self._send(200, body, ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/run":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = json.loads(self.rfile.read(length) or b"{}")
        try:
            params = _parse_params(raw)
            run_with(params, OUTDIR, plot=True)
            summary = (OUTDIR / "summary.txt").read_text(encoding="utf-8")
            figures = sorted(p.name for p in OUTDIR.glob("fig*.png"))
            self._send(200, json.dumps({"summary": summary, "figures": figures}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as exc:
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")


def main(port: int = 8000) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    server = LocalHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"网页界面已启动：{url} （按 Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
